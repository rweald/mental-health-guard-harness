# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Label-audit report: integrity defects fail, judge-quality signals only warn."""

from mh_guard_harness.generation import Dialogue
from mh_guard_harness.label_audit import audit_labels, format_report
from mh_guard_harness.scenarios import Persona, Scenario


def _scn(sid, turn_script, risk="self_harm", sub="direct_ideation"):
    return Scenario(id=sid, risk_category=risk, subcategory=sub, persona=Persona(name="X"),
                    trajectory="escalation", turn_script=turn_script, opening_instruction="o")


def _dlg(sid, tags, seed=0):
    msgs = []
    for i in range(len(tags)):
        msgs.append({"role": "user", "content": f"{sid} u{i}"})
        if i < len(tags) - 1:
            msgs.append({"role": "assistant", "content": f"{sid} a{i}"})
    return Dialogue(scenario_id=sid, seed=seed, messages=msgs, turn_tags=tags)


def _row(did, i, label):
    return {"dialogue_id": did, "user_turn_index": i, "judge_label": label}


def _by_label(rep):
    return {c.label: c for c in rep.checks}


def _clean_setup():
    # one self_harm dialogue (SAFE, UNSAFE) + one safe dialogue (SAFE, BORDERLINE)
    scn = [_scn("a", ["SAFE", "UNSAFE"]),
           _scn("b", ["SAFE", "BORDERLINE"], risk="safe", sub="work")]
    dl = [_dlg("a", ["SAFE", "UNSAFE"]), _dlg("b", ["SAFE", "BORDERLINE"])]
    rows = [
        _row("a#seed0", 1, "safe"), _row("a#seed0", 2, "unsafe_self_harm_risk"),
        _row("b#seed0", 1, "safe"), _row("b#seed0", 2, "safe"),
    ]
    return rows, dl, scn


def test_clean_labels_pass_with_correct_signals():
    rows, dl, scn = _clean_setup()
    rep = audit_labels(rows, dl, scn)
    assert rep.ok and not rep.failures
    chk = _by_label(rep)
    assert chk["coverage"].value.startswith("2/2")
    assert chk["null labels"].value == "0"
    assert chk["UNSAFE recall"].value.startswith("100.0%") and chk["UNSAFE recall"].severity == "ok"
    assert chk["UNSAFE type match"].value.startswith("100.0%")
    assert chk["safe-scenario FPR"].value.startswith("0.00%")


def test_null_label_is_integrity_failure():
    rows, dl, scn = _clean_setup()
    rows[1]["judge_label"] = None  # judge parse produced no vote for a#seed0 turn 2
    rep = audit_labels(rows, dl, scn)
    assert not rep.ok
    assert _by_label(rep)["null labels"].severity == "fail"


def test_partial_and_missing_coverage_fails():
    rows, dl, scn = _clean_setup()
    rows = [r for r in rows if not (r["dialogue_id"] == "a#seed0" and r["user_turn_index"] == 2)]
    rep = audit_labels(rows, dl, scn)  # dialogue 'a' now missing a turn
    assert _by_label(rep)["coverage"].severity == "fail"


def test_duplicate_turn_label_fails():
    rows, dl, scn = _clean_setup()
    rows.append(_row("a#seed0", 1, "safe"))  # same (dialogue, turn) twice
    rep = audit_labels(rows, dl, scn)
    assert _by_label(rep)["duplicate (dialogue, turn)"].severity == "fail"


def test_weak_judge_signals_warn_not_fail():
    # a judge that misses the unsafe turn: low recall + type miss, but data is intact
    rows, dl, scn = _clean_setup()
    rows[1]["judge_label"] = "safe"  # UNSAFE turn labeled safe
    rep = audit_labels(rows, dl, scn)
    chk = _by_label(rep)
    assert chk["UNSAFE recall"].severity == "warn"
    assert chk["UNSAFE type match"].severity == "warn"
    assert rep.ok  # quality signals never fail the audit


def test_safe_fpr_warns_when_safe_turn_called_unsafe():
    rows, dl, scn = _clean_setup()
    rows[3]["judge_label"] = "unsafe_self_harm_risk"  # a safe-scenario turn called unsafe
    rep = audit_labels(rows, dl, scn)
    chk = _by_label(rep)
    assert chk["safe-scenario FPR"].severity == "warn"
    assert rep.ok


def test_signals_skipped_without_library():
    rows, dl, _ = _clean_setup()
    rep = audit_labels(rows, dl, scenarios=None)
    chk = _by_label(rep)
    assert "UNSAFE type match" not in chk  # needs categories
    assert "safe-scenario FPR" not in chk
    assert "UNSAFE recall" in chk  # tag-based, still available
    assert "CLEAN" in format_report(rep) or "FAILURE" in format_report(rep)
