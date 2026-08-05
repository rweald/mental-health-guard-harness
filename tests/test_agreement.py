# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Judge-vs-clinician agreement metrics."""

import math

from mh_guard_harness.agreement import agreement, confusion_3way


def test_perfect_agreement_across_spellings():
    # judge uses threats_to_others; gold uses harm_to_others -> normalize to same class
    judge = ["safe", "unsafe_self_harm_risk", "unsafe_threats_to_others"]
    gold = ["safe", "unsafe_self_harm_risk", "unsafe_harm_to_others"]
    a = agreement(judge, gold)
    assert a["kappa_3way"] == 1.0
    assert a["kappa_binary"] == 1.0
    assert a["accuracy"] == 1.0
    assert a["pos_f1"] == 1.0
    assert a["coverage"] == 1.0


def test_length_mismatch_raises():
    try:
        agreement(["safe"], ["safe", "safe"])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_none_labels_excluded_but_counted_in_coverage():
    judge = ["safe", None, "unsafe_self_harm_risk", None]
    gold = ["safe", "safe", "unsafe_self_harm_risk", "unsafe_harm_to_others"]
    a = agreement(judge, gold)
    assert a["n"] == 4
    assert a["n_scored"] == 2
    assert a["coverage"] == 0.5
    assert a["accuracy"] == 1.0  # the 2 scored ones both agree


def test_all_none_returns_nan_without_crashing():
    a = agreement([None, None], ["safe", "unsafe_self_harm_risk"])
    assert a["n_scored"] == 0
    assert math.isnan(a["kappa_3way"]) and math.isnan(a["pos_f1"])


def test_disagreement_lowers_scores():
    # judge calls everything safe; gold has real positives -> pos_f1 = 0
    judge = ["safe", "safe", "safe", "safe"]
    gold = ["safe", "unsafe_self_harm_risk", "unsafe_harm_to_others", "safe"]
    a = agreement(judge, gold)
    assert a["pos_f1"] == 0.0
    assert a["n_positive_gold"] == 2
    assert a["accuracy"] == 0.5


def test_binary_collapse_ignores_which_unsafe():
    # judge confuses the two unsafe *types* but binary (safe/unsafe) is perfect.
    # Include a safe turn so the binary arrays span both classes (else kappa is nan).
    judge = ["safe", "unsafe_threats_to_others", "unsafe_self_harm_risk"]
    gold = ["safe", "unsafe_self_harm_risk", "unsafe_harm_to_others"]
    a = agreement(judge, gold)
    assert a["kappa_binary"] == 1.0        # safe/unsafe agrees on all three
    assert a["accuracy"] == 1 / 3          # but 3-way wrong on the two unsafe turns
    assert a["kappa_3way"] < a["kappa_binary"]


def test_directional_errors_and_counts():
    # 2 gold unsafe (1 caught, 1 missed) + 2 gold safe (1 flagged, 1 correct)
    judge = ["unsafe_self_harm_risk", "safe", "unsafe_self_harm_risk", "safe"]
    gold = ["unsafe_self_harm_risk", "unsafe_harm_to_others", "safe", "safe"]
    a = agreement(judge, gold)
    assert a["counts"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert a["fnr"] == 0.5   # missed 1 of 2 unsafe
    assert a["fpr"] == 0.5   # false-alarmed 1 of 2 safe
    assert a["recall"] == 0.5
    assert a["precision"] == 0.5


def test_recall_by_category():
    # self_harm: 2 turns, both caught; harm_to_others: 2 turns, 1 caught
    judge = ["unsafe_self_harm_risk", "unsafe_self_harm_risk",
             "unsafe_threats_to_others", "safe"]
    gold = ["unsafe_self_harm_risk", "unsafe_self_harm_risk",
            "unsafe_harm_to_others", "unsafe_harm_to_others"]
    a = agreement(judge, gold)
    assert a["recall_by_category"]["self_harm"] == 1.0
    assert a["recall_by_category"]["harm_to_others"] == 0.5


def test_empty_has_directional_nan_keys():
    a = agreement([None, None], ["safe", "unsafe_self_harm_risk"])
    for k in ("fnr", "fpr", "precision", "recall"):
        assert math.isnan(a[k])
    assert a["counts"] == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def test_confusion_3way_shape_and_counts():
    judge = ["safe", "unsafe_self_harm_risk", "unsafe_threats_to_others", None]
    gold = ["safe", "unsafe_self_harm_risk", "unsafe_self_harm_risk", "safe"]
    m = confusion_3way(judge, gold)
    assert m["safe"]["safe"] == 1
    assert m["self_harm"]["self_harm"] == 1         # correct self_harm
    assert m["self_harm"]["harm_to_others"] == 1    # self_harm judged as harm_to_others
    # the None-judge turn is skipped entirely
    assert sum(m[g][j] for g in m for j in m[g]) == 3
