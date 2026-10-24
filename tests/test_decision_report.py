# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Decision-model comparison report: loading, alignment checks, failure policies,
confusion, Holm adjustment, and an end-to-end build on synthetic files."""

import json

import numpy as np
import pytest

from mh_guard_harness import decision_report as dr

# 8 rows: 0-1 self-harm, 2 harm to others, 3-7 safe.
LABELS = ["unsafe_self_harm_risk", "unsafe_self_harm_risk", "unsafe_harm_to_others",
          "safe", "safe", "safe", "safe", "safe"]
GOOD = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.15, 0.05]


def _write_scores(path, scores, *, labels=LABELS, extra=None, skip=()):
    with open(path, "w") as f:
        for i, (p, lbl) in enumerate(zip(scores, labels)):
            if i in skip:
                continue
            rec = {"index": i, "p_unsafe": p, "label": lbl, "annotator_labels": {}}
            if extra:
                rec.update(extra(i))
            f.write(json.dumps(rec) + "\n")
    return path


def _write_model(d, model, scores, *, failed=(), spec="s" * 64, choices=None):
    for suffix, question in (("choice3", "risk_category"), ("yesno", "unsafe")):
        extra = (lambda i: {"choice": choices[i]}) if choices and suffix == "choice3" else None
        path = _write_scores(d / f"{model}__{suffix}_scores.jsonl", scores,
                             skip=set(failed), extra=extra)
        path.with_suffix(".provenance.json").write_text(json.dumps({
            "question": question,
            "failures": {"refusal": len(failed)} if failed else {},
            "failures_by_class": {},
            "decision_spec": {"sha256": spec},
            "run": {"answered": 8, "cost_usd": 0.004, "latency_s": {"p50": 0.1, "p95": 0.2},
                    "resolved_models": {f"v/{model}-1": 8}, "providers": {"V": 8},
                    "rows_over_long_state_bytes": 0},
        }))
    with open(d / f"{model}.failures.jsonl", "w") as f:
        for q in ("risk_category", "unsafe"):
            for i in failed:
                f.write(json.dumps({"index": i, "question": q, "kind": "refusal",
                                    "label": LABELS[i]}) + "\n")


@pytest.fixture
def setup(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    mg = _write_scores(base / "mg.jsonl", GOOD)
    glm = [_write_scores(base / f"glm{s}.jsonl", [p + 0.01 * s for p in GOOD])
           for s in range(3)]
    dec = tmp_path / "dec"
    dec.mkdir()
    return mg, glm, dec


def _build(setup, **kw):
    mg, glm, dec = setup
    return dr.build(decisions_dir=dec, mindguard_path=mg, glm_paths=glm,
                    n_boot=50, workers=1, **kw)


def test_holm_matches_hand_computation():
    # sorted p: .01*4=.04, .02*3=.06, .03*2=.06 (monotone), .5*1=.5
    assert dr.holm([0.03, 0.01, 0.5, 0.02]) == pytest.approx([0.06, 0.04, 0.5, 0.06])
    out = dr.holm([float("nan"), 0.01])
    assert np.isnan(out[0]) and out[1] == pytest.approx(0.01)


def test_identical_model_to_mindguard_has_zero_delta(setup):
    _write_model(setup[2], "same", GOOD)
    report = _build(setup)
    c = [x for x in report["comparisons"]
         if x["b"] == dr.MINDGUARD and x["scope"] == "overall"]
    assert c and all(x["delta"] == 0 for x in c)
    assert report["models"]["same"]["bootstrap_policies"] == ["as_unsafe"]


def test_failure_policies_differ_and_pair_on_scored_rows(setup):
    _write_model(setup[2], "refuser", GOOD, failed=[3])  # refuses a safe row
    report = _build(setup)
    m = report["models"]["refuser"]
    assert m["bootstrap_policies"] == ["excluded", "as_unsafe"]
    excl = m["questions"]["yesno"]["metrics"]["excluded"]
    fill = m["questions"]["yesno"]["metrics"]["as_unsafe"]
    assert excl["n"] == 7 and fill["n"] == 8
    assert excl["auroc"] == 1.0
    assert fill["auroc"] < 1.0  # the safe row scored 1.0 now outranks the positives
    vs = [x for x in report["comparisons"] if x["a"] == "refuser / yesno"
          and x["b"] == dr.MINDGUARD and x["scope"] == "overall"]
    assert {x["policy"]: x["n"] for x in vs} == {"excluded": 7, "as_unsafe": 8}


def test_confusion_counts_gold_against_choice(setup):
    choices = ["self_harm", "safe", "self_harm", "safe", "safe", "safe", "safe",
               "harm_to_others"]
    _write_model(setup[2], "m", GOOD, choices=choices)
    conf = _build(setup)["models"]["m"]["questions"]["choice3"]["confusion"]
    assert conf["self_harm"] == {"safe": 1, "self_harm": 1, "harm_to_others": 0}
    assert conf["harm_to_others"]["self_harm"] == 1
    assert conf["safe"] == {"safe": 4, "self_harm": 0, "harm_to_others": 1}


def test_different_specs_are_refused(setup):
    _write_model(setup[2], "a", GOOD, spec="a" * 64)
    _write_model(setup[2], "b", GOOD, spec="b" * 64)
    with pytest.raises(ValueError, match="different"):
        _build(setup)


def test_rows_missing_from_both_scores_and_failures_are_refused(setup):
    _write_model(setup[2], "m", GOOD)
    path = setup[2] / "m__yesno_scores.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")  # drop a row with no failure entry
    with pytest.raises(ValueError, match="partition"):
        _build(setup)


def test_misaligned_labels_are_refused(setup):
    mg, glm, dec = setup
    _write_scores(glm[1], GOOD, labels=list(reversed(LABELS)))
    _write_model(dec, "m", GOOD)
    with pytest.raises(ValueError, match="disagree with gold"):
        _build(setup)


def test_limit_files_are_ignored_and_order_is_fixed(setup):
    dec = setup[2]
    for m in ("zeta", "clef", "jev-1.14"):
        _write_model(dec, m, GOOD)
    (dec / "clef__limit5__choice3_scores.jsonl").write_text("")
    assert dr.find_models(dec) == ["jev-1.14", "clef", "zeta"]


def test_glm_seed_summary_and_render(setup):
    _write_model(setup[2], "m", GOOD)
    report = _build(setup)
    glm = report["baselines"][dr.GLM]
    assert len(glm["per_seed_auroc"]) == 3
    text = dr.render(report)
    assert "no model had failures" in text
    assert "m / choice3" in text and "Holm" in text
