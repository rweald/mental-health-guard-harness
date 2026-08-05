# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Inferential comparisons for the judge-agreement study.

The testset has only 42 positives and the AUROC ceiling is high, so point estimates
are soft — a judge difference may be noise. These add the uncertainty layer:
- ``bootstrap_kappa_ci``: a percentile CI on a judge's κ vs the gold.
- ``mcnemar_compare``: a paired significance test on whether one judge agrees with the
  gold more than another, over the turns both labeled (same turns → paired, not two
  independent accuracies).
- ``context_effect``: the full-vs-turn κ gap for one judge (how much it leans on future
  context).
- ``bootstrap_auroc_diff``: a paired CI on the AUROC gap between two fine-tuned arms
  — the same-rows pairing is what makes a sub-0.01 difference testable at all.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

from .agreement import agreement
from .metrics import auroc, cohen_kappa, risk_category

_ANNOTATORS = ("annotator_1", "annotator_2", "annotator_3")


def _cats(labels):
    return [risk_category(x) for x in labels]


def _labels_for(pairs, which):
    if which == "binary":
        return np.array([0 if risk_category(x) == "safe" else 1 for x in pairs])
    return np.array([risk_category(x) for x in pairs], dtype=object)


def bootstrap_kappa_ci(
    judge_labels,
    gold_labels,
    *,
    which: str = "3way",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI on Cohen's κ (``which`` = "3way" or "binary") by resampling turns.
    Degenerate resamples (a single class present → κ undefined) are dropped."""
    pairs = [(j, g) for j, g in zip(judge_labels, gold_labels) if j is not None]
    if len(pairs) < 2:
        return (float("nan"), float("nan"))
    judge = _labels_for([j for j, _ in pairs], which)
    gold = _labels_for([g for _, g in pairs], which)

    rng = np.random.default_rng(seed)
    n = len(pairs)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        k = cohen_kappa(list(judge[idx]), list(gold[idx]))
        if not math.isnan(k):
            stats.append(k)
    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def bootstrap_auroc_diff(
    y_true,
    p_a,
    p_b,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Paired percentile CI on ``AUROC(p_a) - AUROC(p_b)`` over the same rows.

    Both arms score the identical testset, so resampling *rows* (not arms) keeps the
    comparison paired — the shared difficulty of a given turn cancels, which is the only
    reason a sub-0.01 AUROC gap is testable with 42 positives at all.

    The resample is **stratified**: positives and negatives are drawn separately, at
    their original counts. An unstratified draw would let a replicate land with very few
    positives (or none), inflating the variance with a class-balance wobble that the real
    fixed testset does not have.

    Returns ``{"delta", "ci_low", "ci_high", "pvalue", "n_boot"}``. ``pvalue`` is the
    two-sided percentile-bootstrap value — twice the smaller tail mass on either side of
    zero, floored at ``1 / n_boot`` (the bootstrap cannot resolve below its own
    resolution, so never report 0) and capped at 1.0 (doubling overshoots when the
    replicates pile up on zero, as they do for two identical arms). All NaN when a class
    is absent.
    """
    y_true = np.asarray(y_true)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    nan = {
        "delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
        "pvalue": float("nan"), "n_boot": 0,
    }
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    if len(pos) == 0 or len(neg) == 0:
        return nan

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        d = auroc(y_true[idx], p_a[idx]) - auroc(y_true[idx], p_b[idx])
        if not math.isnan(d):
            deltas.append(d)
    if not deltas:
        return nan

    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    tail = min(float((deltas <= 0).mean()), float((deltas >= 0).mean()))
    return {
        "delta": auroc(y_true, p_a) - auroc(y_true, p_b),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "pvalue": min(1.0, max(2 * tail, 1 / len(deltas))),
        "n_boot": len(deltas),
    }


def mcnemar_compare(judge_a, judge_b, gold_labels) -> dict:
    """Paired McNemar test on per-turn *correctness* (3-way match to gold) between two
    judges, over turns both labeled. ``a_only``/``b_only`` are the discordant counts;
    a small p-value means one judge is reliably more often right than the other."""
    a_correct, b_correct = [], []
    for ja, jb, g in zip(judge_a, judge_b, gold_labels):
        if ja is None or jb is None:
            continue
        gc = risk_category(g)
        a_correct.append(risk_category(ja) == gc)
        b_correct.append(risk_category(jb) == gc)
    a = np.array(a_correct)
    b = np.array(b_correct)
    a_only = int((a & ~b).sum())  # a right, b wrong
    b_only = int((~a & b).sum())  # b right, a wrong
    both = int((a & b).sum())
    neither = int((~a & ~b).sum())
    if a_only + b_only == 0:
        pvalue = 1.0  # no discordant pairs -> nothing to distinguish
    else:
        # exact binomial when the discordant count is small, else the chi-square form
        res = mcnemar([[both, a_only], [b_only, neither]], exact=(a_only + b_only < 25))
        pvalue = float(res.pvalue)
    return {"n": int(len(a)), "a_only": a_only, "b_only": b_only, "pvalue": pvalue}


def annotator_pairwise_kappa(annotators) -> dict:
    """Pairwise Cohen κ among the 3 clinician annotators (3-way) — the human agreement
    band a judge is measured against. ``annotators`` is a list of per-turn dicts
    ``{annotator_1, annotator_2, annotator_3}``."""
    per = {n: _cats([a[n] for a in annotators]) for n in _ANNOTATORS}
    pairwise = {
        f"{x[-1]}_vs_{y[-1]}": cohen_kappa(per[x], per[y])
        for x, y in combinations(_ANNOTATORS, 2)
    }
    vals = [v for v in pairwise.values() if not math.isnan(v)]
    return {
        "pairwise": pairwise,
        "min": min(vals) if vals else float("nan"),
        "max": max(vals) if vals else float("nan"),
        "mean": float(np.mean(vals)) if vals else float("nan"),
    }


def judge_vs_annotators_kappa(judge_labels, annotators) -> dict:
    """Cohen κ between the judge and each individual annotator (3-way), over turns the
    judge labeled — the judge treated as a 4th rater. Compare ``mean`` to the human band
    from ``annotator_pairwise_kappa``: inside it means clinician-indistinguishable."""
    rows = [(j, a) for j, a in zip(judge_labels, annotators) if j is not None]
    if len(rows) < 2:
        return {"per": {f"judge_vs_{n[-1]}": float("nan") for n in _ANNOTATORS},
                "mean": float("nan")}
    judge = _cats([j for j, _ in rows])
    per = {
        f"judge_vs_{n[-1]}": cohen_kappa(judge, _cats([a[n] for _, a in rows]))
        for n in _ANNOTATORS
    }
    vals = [v for v in per.values() if not math.isnan(v)]
    return {"per": per, "mean": float(np.mean(vals)) if vals else float("nan")}


def context_effect(judge_full, judge_turn, gold_labels) -> dict:
    """Full-minus-turn κ for one judge: how much full-conversation context (future
    turns) changes its agreement with the gold. Positive = future context helps."""
    kf = agreement(judge_full, gold_labels)["kappa_3way"]
    kt = agreement(judge_turn, gold_labels)["kappa_3way"]
    return {"kappa_full": kf, "kappa_turn": kt, "delta": kf - kt}
