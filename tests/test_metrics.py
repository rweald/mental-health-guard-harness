# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Tests for the metrics module: pure array math (AUROC / ROC / FPR@TPR / kappa)
with analytically-known answers, degenerate-case guards, and a light JSONL round-trip
through ``load_scores`` -> ``evaluate``.

No model, no GPU, no dataset — every input is hand-constructed so the expected number
is known in closed form (or cross-checked against sklearn on the spot).
"""

import json
import math

import numpy as np
from sklearn.metrics import roc_auc_score

from mh_guard_harness import metrics


# --- binary_labels / risk_category ----------------------------------------------


def test_binary_labels_all_three_strings():
    labels = ["safe", "unsafe_self_harm_risk", "unsafe_harm_to_others"]
    assert list(metrics.binary_labels(labels)) == [0, 1, 1]


def test_binary_labels_unknown_label_is_positive():
    assert list(metrics.binary_labels(["safe", "something_else"])) == [0, 1]


def test_risk_category_both_harm_spellings_map_together():
    assert metrics.risk_category("unsafe_harm_to_others") == "harm_to_others"
    assert metrics.risk_category("unsafe_threats_to_others") == "harm_to_others"
    assert metrics.risk_category("unsafe_self_harm_risk") == "self_harm"
    assert metrics.risk_category("safe") == "safe"
    assert metrics.risk_category("mystery") == "unknown"


# --- auroc ----------------------------------------------------------------------


def test_auroc_perfect_separation_is_one():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    assert metrics.auroc(y, p) == 1.0


def test_auroc_reversed_is_zero():
    y = [0, 0, 1, 1]
    p = [0.9, 0.8, 0.2, 0.1]
    assert metrics.auroc(y, p) == 0.0


def test_auroc_matches_sklearn_on_small_case():
    y = [0, 1, 0, 1, 1, 0, 0, 1]
    p = [0.2, 0.6, 0.55, 0.9, 0.3, 0.1, 0.65, 0.8]
    assert metrics.auroc(y, p) == roc_auc_score(y, p)


def test_auroc_all_ties_is_half():
    # No ranking information -> chance-level 0.5.
    assert metrics.auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auroc_no_positives_is_nan_no_crash():
    assert math.isnan(metrics.auroc([0, 0, 0, 0], [0.1, 0.9, 0.4, 0.5]))


def test_auroc_no_negatives_is_nan():
    assert math.isnan(metrics.auroc([1, 1, 1], [0.1, 0.9, 0.4]))


def test_auroc_empty_is_nan():
    assert math.isnan(metrics.auroc([], []))


# --- roc_sweep ------------------------------------------------------------------


def test_roc_sweep_monotonic_with_corner_endpoints():
    y = [0, 0, 1, 1, 0, 1, 0, 1]
    p = [0.1, 0.35, 0.4, 0.8, 0.2, 0.9, 0.3, 0.6]
    fpr, tpr, thresholds = metrics.roc_sweep(y, p)

    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == 1.0 and tpr[-1] == 1.0
    assert np.all(np.diff(fpr) >= 0)
    assert np.all(np.diff(tpr) >= 0)
    # thresholds run high -> low.
    assert np.all(np.diff(thresholds) <= 0)


# --- fpr_at_tpr -----------------------------------------------------------------


def _analytic_case():
    """20 positives spread 0.30..0.49, 10 negatives with known counts above the
    90%/95% operating thresholds.

    - 90% TPR (18/20) -> threshold 0.32; negatives >= 0.32: just {0.325} -> FPR 0.1.
    - 95% TPR (19/20) -> threshold 0.31; negatives >= 0.31: {0.325, 0.315} -> FPR 0.2.
    """
    pos = [round(0.30 + 0.01 * i, 2) for i in range(20)]
    neg = [0.325, 0.315] + [0.005] * 8
    p = pos + neg
    y = [1] * len(pos) + [0] * len(neg)
    return y, p


def test_fpr_at_tpr_90_analytic():
    y, p = _analytic_case()
    r = metrics.fpr_at_tpr(y, p, 0.90)
    assert r["achieved_tpr"] == 0.90
    assert r["fpr"] == 0.1
    assert r["threshold"] == 0.32


def test_fpr_at_tpr_95_analytic():
    y, p = _analytic_case()
    r = metrics.fpr_at_tpr(y, p, 0.95)
    assert r["achieved_tpr"] == 0.95
    assert r["fpr"] == 0.2
    assert r["threshold"] == 0.31


def test_fpr_at_90_95_convenience_bundle():
    y, p = _analytic_case()
    out = metrics.fpr_at_90_95(y, p)
    assert out["fpr_at_90"]["fpr"] == 0.1
    assert out["fpr_at_95"]["fpr"] == 0.2


def test_fpr_at_tpr_smallest_achieved_tpr_at_or_above_target():
    # 10 positives -> TPR grid is 0.1-steps; target 0.85 rounds up to 0.9.
    pos = [0.5 + 0.01 * i for i in range(10)]
    neg = [0.1] * 10
    y = [1] * 10 + [0] * 10
    p = pos + neg
    r = metrics.fpr_at_tpr(y, p, 0.85)
    assert r["achieved_tpr"] == 0.9


def test_fpr_at_tpr_unreachable_target_is_nan():
    y, p = _analytic_case()
    r = metrics.fpr_at_tpr(y, p, 1.5)
    assert math.isnan(r["fpr"])
    assert math.isnan(r["threshold"])
    assert math.isnan(r["achieved_tpr"])


def test_fpr_at_tpr_no_positives_is_nan():
    r = metrics.fpr_at_tpr([0, 0, 0], [0.1, 0.5, 0.9], 0.9)
    assert math.isnan(r["fpr"])


# --- subgroup_auroc -------------------------------------------------------------


def test_subgroup_auroc_isolates_one_category_vs_safe():
    # Ranking: self_harm scores high, safe low, harm_to_others deliberately LOW so it
    # would tank a naive AUROC — but subgroup self_harm ignores it entirely -> 1.0.
    risk = np.array(["safe", "safe", "self_harm", "harm_to_others", "harm_to_others"])
    p = np.array([0.1, 0.2, 0.9, 0.05, 0.02])
    assert metrics.subgroup_auroc(risk, p, "self_harm") == 1.0
    # harm_to_others ranks BELOW safe here -> reversed -> 0.0.
    assert metrics.subgroup_auroc(risk, p, "harm_to_others") == 0.0


def test_subgroup_auroc_absent_subgroup_is_nan():
    risk = np.array(["safe", "safe", "self_harm"])
    p = np.array([0.1, 0.2, 0.9])
    assert math.isnan(metrics.subgroup_auroc(risk, p, "harm_to_others"))


# --- kappa ----------------------------------------------------------------------


def test_fleiss_kappa_perfect_agreement_is_one():
    # Two categories present, all raters agree per item -> 1.0.
    data = [["a", "a", "a", "a"], ["b", "b", "b", "b"], ["a", "a", "a", "a"]]
    assert metrics.fleiss_kappa_annotators(data) == 1.0


def test_fleiss_kappa_known_reference_value():
    # 3 items x 4 raters: rows [4,0], [2,2], [0,4] -> Pbar=7/9, Pe=1/2 -> kappa=5/9.
    data = [
        ["a", "a", "a", "a"],
        ["a", "a", "b", "b"],
        ["b", "b", "b", "b"],
    ]
    assert metrics.fleiss_kappa_annotators(data) == round(5 / 9, 10) or math.isclose(
        metrics.fleiss_kappa_annotators(data), 5 / 9, rel_tol=1e-9
    )


def test_fleiss_kappa_accepts_list_of_dicts():
    data = [
        {"annotator_1": "a", "annotator_2": "a", "annotator_3": "a"},
        {"annotator_1": "b", "annotator_2": "b", "annotator_3": "b"},
    ]
    assert metrics.fleiss_kappa_annotators(data) == 1.0


def test_fleiss_kappa_single_category_is_nan():
    data = [["a", "a", "a"], ["a", "a", "a"]]
    assert math.isnan(metrics.fleiss_kappa_annotators(data))


def test_cohen_kappa_perfect_agreement_is_one():
    a = ["safe", "unsafe_self_harm_risk", "safe", "unsafe_harm_to_others"]
    assert metrics.cohen_kappa(a, list(a)) == 1.0


def test_cohen_kappa_chance_agreement_is_zero():
    # 100 items, marginals 50/50 on both raters, agreement == chance -> kappa 0.
    a = [1] * 50 + [0] * 50
    b = [1] * 25 + [0] * 25 + [1] * 25 + [0] * 25
    assert metrics.cohen_kappa(a, b) == 0.0


# --- load_scores + evaluate integration -----------------------------------------


def _fixture_rows():
    """~10 rows mixing safe / self_harm / harm_to_others, with a clean ranking so
    AUROC is a comfortable 1.0-ish and subgroups are well-defined."""
    rows = []
    # 6 safe (low scores).
    for i in range(6):
        rows.append(
            {
                "index": i,
                "p_unsafe": 0.05 + 0.02 * i,
                "label": "safe",
                "annotator_labels": {
                    "annotator_1": "safe",
                    "annotator_2": "safe",
                    "annotator_3": "safe",
                },
            }
        )
    # 2 self_harm (high scores).
    for j, idx in enumerate(range(6, 8)):
        rows.append(
            {
                "index": idx,
                "p_unsafe": 0.85 + 0.05 * j,
                "label": "unsafe_self_harm_risk",
                "annotator_labels": {
                    "annotator_1": "unsafe_self_harm_risk",
                    "annotator_2": "unsafe_self_harm_risk",
                    "annotator_3": "safe" if j == 0 else "unsafe_self_harm_risk",
                },
            }
        )
    # 2 harm_to_others (high scores).
    for j, idx in enumerate(range(8, 10)):
        rows.append(
            {
                "index": idx,
                "p_unsafe": 0.80 + 0.05 * j,
                "label": "unsafe_harm_to_others",
                "annotator_labels": {
                    "annotator_1": "unsafe_harm_to_others",
                    "annotator_2": "unsafe_harm_to_others",
                    "annotator_3": "unsafe_harm_to_others",
                },
            }
        )
    return rows


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_load_scores_shapes_and_derived_columns(tmp_path):
    path = tmp_path / "scores.jsonl"
    _write_jsonl(path, _fixture_rows())
    df = metrics.load_scores(path)

    assert len(df) == 10
    assert set(["p_unsafe", "label", "y_true", "risk_category"]).issubset(df.columns)
    assert list(df["y_true"]) == [0] * 6 + [1] * 4
    assert set(df["risk_category"]) == {"safe", "self_harm", "harm_to_others"}
    # annotator columns preserved.
    assert {"annotator_1", "annotator_2", "annotator_3"}.issubset(df.columns)


def test_load_scores_tolerates_blank_lines(tmp_path):
    path = tmp_path / "scores.jsonl"
    rows = _fixture_rows()
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n")
    assert len(metrics.load_scores(path)) == 10


def test_evaluate_keys_and_sane_values(tmp_path):
    path = tmp_path / "scores.jsonl"
    _write_jsonl(path, _fixture_rows())
    report = metrics.evaluate(metrics.load_scores(path))

    expected_keys = {
        "n",
        "n_positive",
        "n_negative",
        "auroc",
        "fpr_at_90",
        "fpr_at_95",
        "subgroup_auroc",
        "fleiss_kappa",
        "cohen_kappa_pairwise",
    }
    assert expected_keys.issubset(report.keys())

    assert report["n"] == 10
    assert report["n_positive"] == 4
    assert report["n_negative"] == 6
    # All unsafe rows out-score all safe rows -> perfect ranking.
    assert report["auroc"] == 1.0
    assert report["subgroup_auroc"]["self_harm"] == 1.0
    assert report["subgroup_auroc"]["harm_to_others"] == 1.0
    assert 0.0 <= report["fpr_at_90"]["fpr"] <= 1.0
    assert -1.0 <= report["fleiss_kappa"] <= 1.0
    assert report["cohen_kappa_pairwise"]  # non-empty for 3 annotators


def test_evaluate_accepts_path_directly(tmp_path):
    path = tmp_path / "scores.jsonl"
    _write_jsonl(path, _fixture_rows())
    report = metrics.evaluate(path)
    assert report["n"] == 10


def test_evaluate_all_safe_slice_auroc_nan(tmp_path):
    # Degenerate: no positives at all -> AUROC/subgroups NaN, but no crash and counts ok.
    rows = [
        {
            "index": i,
            "p_unsafe": 0.1 * i,
            "label": "safe",
            "annotator_labels": {
                "annotator_1": "safe",
                "annotator_2": "safe",
                "annotator_3": "safe",
            },
        }
        for i in range(4)
    ]
    path = tmp_path / "safe.jsonl"
    _write_jsonl(path, rows)
    report = metrics.evaluate(path)

    assert report["n_positive"] == 0
    assert math.isnan(report["auroc"])
    assert math.isnan(report["subgroup_auroc"]["self_harm"])
    assert math.isnan(report["fpr_at_90"]["fpr"])
