# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Inferential comparisons: bootstrap CI, paired McNemar, context effect, arm AUROC gap."""

import math

import numpy as np

from mh_guard_harness.analysis import (
    annotator_pairwise_kappa,
    bootstrap_auroc_diff,
    bootstrap_kappa_ci,
    context_effect,
    judge_vs_annotators_kappa,
    mcnemar_compare,
)


def _ann(a1, a2, a3):
    return {"annotator_1": a1, "annotator_2": a2, "annotator_3": a3}


def _mix(n_safe, n_unsafe, judge_wrong=0):
    """Build parallel judge/gold lists: n_safe safe turns + n_unsafe self_harm turns,
    with the first ``judge_wrong`` unsafe turns mislabeled safe by the judge."""
    gold = ["safe"] * n_safe + ["unsafe_self_harm_risk"] * n_unsafe
    judge = ["safe"] * n_safe + (
        ["safe"] * judge_wrong + ["unsafe_self_harm_risk"] * (n_unsafe - judge_wrong)
    )
    return judge, gold


def test_bootstrap_ci_brackets_perfect_agreement():
    judge, gold = _mix(20, 20)  # perfect agreement -> kappa 1.0
    lo, hi = bootstrap_kappa_ci(judge, gold, n_boot=500, seed=1)
    assert 0.9 <= lo <= 1.0 and hi == 1.0


def test_bootstrap_ci_wider_when_noisier():
    perfect = bootstrap_kappa_ci(*_mix(30, 30), n_boot=800, seed=2)
    noisy = bootstrap_kappa_ci(*_mix(30, 30, judge_wrong=12), n_boot=800, seed=2)
    perfect_width = perfect[1] - perfect[0]
    noisy_width = noisy[1] - noisy[0]
    assert noisy_width > perfect_width  # more disagreement -> wider CI


def test_bootstrap_ci_too_few_is_nan():
    lo, hi = bootstrap_kappa_ci(["safe"], ["safe"], n_boot=100)
    assert math.isnan(lo) and math.isnan(hi)


def _ranked(n_safe, n_unsafe, *, sep, seed=0):
    """Scores for one arm: negatives near 0, positives lifted by ``sep``. Larger sep ->
    better separation -> higher AUROC."""
    rng = np.random.default_rng(seed)
    y = np.array([0] * n_safe + [1] * n_unsafe)
    p = rng.uniform(0, 0.5, size=len(y)) + y * sep
    return y, p


def test_auroc_diff_detects_a_real_gap():
    y, better = _ranked(200, 40, sep=0.6)
    _, worse = _ranked(200, 40, sep=0.15)
    r = bootstrap_auroc_diff(y, better, worse, n_boot=400, seed=1)
    assert r["delta"] > 0
    assert r["ci_low"] > 0  # CI clears zero -> the gap is not noise
    assert r["pvalue"] < 0.05


def test_auroc_diff_identical_arms_straddles_zero():
    y, p = _ranked(200, 40, sep=0.4)
    r = bootstrap_auroc_diff(y, p, p, n_boot=400, seed=1)
    assert r["delta"] == 0
    assert r["ci_low"] <= 0 <= r["ci_high"]
    assert r["pvalue"] == 1.0


def test_auroc_diff_pvalue_floored_at_bootstrap_resolution():
    """A p-value below 1/n_boot is unresolvable, so it must not be reported as 0."""
    y, better = _ranked(200, 40, sep=5.0)  # near-perfect separation
    _, worse = _ranked(200, 40, sep=0.0)   # chance
    r = bootstrap_auroc_diff(y, better, worse, n_boot=200, seed=1)
    assert r["pvalue"] == 1 / 200


def test_auroc_diff_is_reproducible_for_a_seed():
    y, a = _ranked(150, 30, sep=0.5)
    _, b = _ranked(150, 30, sep=0.3, seed=7)
    first = bootstrap_auroc_diff(y, a, b, n_boot=300, seed=3)
    again = bootstrap_auroc_diff(y, a, b, n_boot=300, seed=3)
    assert first == again


def test_auroc_diff_no_positives_is_nan():
    y = np.zeros(20, dtype=int)
    r = bootstrap_auroc_diff(y, np.linspace(0, 1, 20), np.linspace(1, 0, 20), n_boot=50)
    assert math.isnan(r["delta"]) and math.isnan(r["pvalue"])


def test_mcnemar_detects_one_judge_better():
    gold = ["unsafe_self_harm_risk"] * 30
    a = ["unsafe_self_harm_risk"] * 30            # A right on all
    b = ["safe"] * 30                             # B wrong on all
    out = mcnemar_compare(a, b, gold)
    assert out["a_only"] == 30 and out["b_only"] == 0
    assert out["pvalue"] < 0.05                   # A reliably better


def test_mcnemar_no_discordant_pairs_is_p1():
    gold = ["safe", "unsafe_self_harm_risk"]
    out = mcnemar_compare(["safe", "safe"], ["safe", "safe"], gold)  # identical judges
    assert out["a_only"] == 0 and out["b_only"] == 0
    assert out["pvalue"] == 1.0


def test_mcnemar_skips_unlabeled_turns():
    gold = ["safe", "safe", "safe"]
    out = mcnemar_compare(["safe", None, "safe"], ["safe", "safe", None], gold)
    assert out["n"] == 1  # only the first turn has both labels


def test_annotator_pairwise_kappa_perfect_and_spelling():
    # all three annotators identical (across harm spellings) -> kappa 1.0 each pair
    anns = [_ann("safe", "safe", "safe"),
            _ann("unsafe_harm_to_others", "unsafe_harm_to_others", "unsafe_threats_to_others"),
            _ann("unsafe_self_harm_risk", "unsafe_self_harm_risk", "unsafe_self_harm_risk")]
    out = annotator_pairwise_kappa(anns)
    assert set(out["pairwise"]) == {"1_vs_2", "1_vs_3", "2_vs_3"}
    assert out["min"] == 1.0 and out["max"] == 1.0


def test_judge_inside_vs_outside_human_band():
    # annotators mostly agree; a judge matching them lands high, a random-ish judge low
    anns, judge_good, judge_bad = [], [], []
    seq = ["safe", "unsafe_self_harm_risk"] * 15
    for i, g in enumerate(seq):
        # annotator 3 disagrees on a few, creating a realistic human band < 1.0
        a3 = g if i % 7 else "safe"
        anns.append(_ann(g, g, a3))
        judge_good.append(g)                       # judge tracks the annotators
        judge_bad.append("safe")                   # judge always safe
    human = annotator_pairwise_kappa(anns)
    good = judge_vs_annotators_kappa(judge_good, anns)["mean"]
    bad = judge_vs_annotators_kappa(judge_bad, anns)["mean"]
    assert good >= human["min"]        # good judge sits in/above the human band
    assert bad < human["min"]          # always-safe judge falls below it


def test_judge_vs_annotators_skips_unlabeled():
    anns = [_ann("safe", "safe", "safe"), _ann("safe", "safe", "safe")]
    out = judge_vs_annotators_kappa(["safe", None], anns)
    # only 1 usable turn -> too few for kappa -> nan mean, but no crash
    assert math.isnan(out["mean"])


def test_context_effect_delta_positive_when_full_better():
    gold = ["safe", "unsafe_self_harm_risk"] * 10
    full = list(gold)                              # full mode perfect
    turn = ["safe"] * 20                           # turn mode misses all unsafe
    ce = context_effect(full, turn, gold)
    assert ce["kappa_full"] > ce["kappa_turn"]
    assert ce["delta"] > 0
