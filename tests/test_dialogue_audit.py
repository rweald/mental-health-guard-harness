# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Dialogue-audit report: clean sets pass, each defect surfaces at the right severity."""

from mh_guard_harness.dialogue_audit import audit_dialogues, format_report
from mh_guard_harness.generation import Dialogue
from mh_guard_harness.scenarios import Persona, Scenario


def _scn(sid, turn_script, risk="self_harm", sub="direct_ideation"):
    return Scenario(
        id=sid, risk_category=risk, subcategory=sub, persona=Persona(name="X"),
        trajectory="escalation", turn_script=turn_script,
        opening_instruction="open",
    )


def _dlg(sid, user_turns, tags, seed=0, flags=None):
    """Build a user-first alternating dialogue ending on a patient turn."""
    msgs = []
    for i, t in enumerate(user_turns):
        msgs.append({"role": "user", "content": t})
        if i < len(user_turns) - 1:
            msgs.append({"role": "assistant", "content": f"reply {sid}-{i}"})
    return Dialogue(scenario_id=sid, seed=seed, messages=msgs, turn_tags=tags,
                    flags=flags or [])


def _by_label(report):
    return {c.label: c for c in report.checks}


def test_clean_set_passes():
    scn = [_scn("a", ["SAFE", "UNSAFE"]), _scn("b", ["SAFE", "BORDERLINE", "UNSAFE"])]
    dl = [
        _dlg("a", ["hi there", "I have pills set aside and a plan"], ["SAFE", "UNSAFE"]),
        _dlg("b", ["hello", "feeling low lately", "I want to end it tonight"],
             ["SAFE", "BORDERLINE", "UNSAFE"]),
    ]
    rep = audit_dialogues(dl, scn)
    assert rep.ok and not rep.failures
    chk = _by_label(rep)
    assert chk["coverage"].value.startswith("2/2")
    assert chk["empty patient turns"].value == "0"
    # UNSAFE-turn stats appear when a library is supplied
    assert "UNSAFE turns" in chk


def test_missing_scenario_is_a_failure():
    scn = [_scn("a", ["SAFE", "UNSAFE"]), _scn("b", ["SAFE", "UNSAFE"])]
    dl = [_dlg("a", ["hi", "plan"], ["SAFE", "UNSAFE"])]  # 'b' never generated
    rep = audit_dialogues(dl, scn)
    assert not rep.ok
    assert _by_label(rep)["coverage"].severity == "fail"


def test_empty_turn_and_bad_alternation_fail():
    dl = [_dlg("a", ["hi", "   "], ["SAFE", "UNSAFE"])]  # second turn blank
    rep = audit_dialogues(dl)  # no library -> coverage skipped, structure still checked
    chk = _by_label(rep)
    assert chk["empty patient turns"].severity == "fail"
    assert chk["coverage"].value.startswith("skipped")


def test_duplicate_key_is_a_failure():
    scn = [_scn("a", ["SAFE", "UNSAFE"])]
    dl = [_dlg("a", ["hi", "x"], ["SAFE", "UNSAFE"]),
          _dlg("a", ["hi2", "y"], ["SAFE", "UNSAFE"])]  # same scenario+seed twice
    rep = audit_dialogues(dl, scn)
    assert _by_label(rep)["coverage"].severity == "fail"


def test_refusal_and_flag_are_warnings_not_failures():
    dl = [_dlg("a", ["I'm sorry, I can't continue this.", "still here"],
               ["SAFE", "UNSAFE"], flags=[{"turn": 1, "reason": "possible_refusal"}])]
    rep = audit_dialogues(dl)
    chk = _by_label(rep)
    assert chk["refusal-looking turns"].severity == "warn"
    assert chk["dialogues flagged (refusal)"].severity == "warn"
    assert rep.ok  # warnings don't fail the audit


def test_shared_openings_and_repeats_flagged():
    dl = [
        _dlg("a", ["same open", "tail a"], ["SAFE", "UNSAFE"]),
        _dlg("b", ["same open", "tail b"], ["SAFE", "UNSAFE"], seed=1),  # shared opening
        _dlg("c", ["loop", "loop"], ["SAFE", "UNSAFE"], seed=2),  # intra-dialogue repeat
    ]
    # give a/b/c distinct scenario ids so coverage is clean
    dl[1].scenario_id = "b"
    dl[2].scenario_id = "c"
    rep = audit_dialogues(dl)
    chk = _by_label(rep)
    assert chk["shared openings"].severity == "warn"
    assert chk["intra-dialogue repeats"].severity == "fail"


def test_format_report_renders_verdict():
    rep = audit_dialogues([_dlg("a", ["hi", "plan"], ["SAFE", "UNSAFE"])])
    out = format_report(rep, title="dialogue audit")
    assert "dialogue audit" in out
    assert "CLEAN" in out or "FAILURE" in out
