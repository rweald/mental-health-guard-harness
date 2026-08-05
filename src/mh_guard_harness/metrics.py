# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Metrics: (scores + gold labels) -> the numbers that validate the harness.

This is the "scores + labels -> numbers" half of evaluation. It reads the seam-contract
JSONL written by ``modeling.scoring`` and turns ``p_unsafe`` + gold labels into the
headline figures: AUROC, an ROC sweep, FPR@{90,95}%TPR, per-risk-category subgroup
AUROC, and inter-annotator agreement (the human ceiling).

Torch-free by design (numpy / scikit-learn / statsmodels / pandas only) so the whole
module unit-tests on CPU without the 8B model, a GPU, or the dataset.

The metric is **binary**: the classifier emits one safe/unsafe token, scored as
``p_unsafe in [0, 1]``. Gold is 3-way but collapses to binary — positive is
``label != "safe"``. The 3-way structure survives only for subgroup AUROC (one number
for self-harm positives vs. safe, one for harm-to-others vs. safe).
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

# Gold label -> risk-category subgroup. Defensive on the harm-to-others spelling:
# the paper's judge emits ``unsafe_threats_to_others`` while the released testset gold
# uses ``unsafe_harm_to_others`` — both collapse to the same subgroup.
_RISK_CATEGORY = {
    "safe": "safe",
    "unsafe_self_harm_risk": "self_harm",
    "unsafe_harm_to_others": "harm_to_others",
    "unsafe_threats_to_others": "harm_to_others",
}

SELF_HARM = "self_harm"
HARM_TO_OTHERS = "harm_to_others"
SAFE = "safe"


def risk_category(label: str) -> str:
    """Map a gold label string to its risk-category subgroup.

    ``safe`` -> ``"safe"``; self-harm -> ``"self_harm"``; either harm-to-others
    spelling -> ``"harm_to_others"``. Anything unrecognized -> ``"unknown"`` (it still
    counts as a binary positive via :func:`binary_labels`, just not attributed to a
    named subgroup).
    """
    return _RISK_CATEGORY.get(label, "unknown")


def binary_labels(labels) -> np.ndarray:
    """Collapse gold labels to the binary positive class: 0 for ``"safe"``, 1 else."""
    return np.array([0 if lbl == SAFE else 1 for lbl in labels], dtype=int)


def load_scores(path) -> pd.DataFrame:
    """Read the seam-contract JSONL (one object per line) into a DataFrame.

    Each input row is ``{"index", "p_unsafe", "label", "annotator_labels"}``. The
    returned frame adds the derived columns the metrics need:

    - ``y_true`` — binary gold (0 safe / 1 unsafe),
    - ``risk_category`` — ``safe`` / ``self_harm`` / ``harm_to_others`` / ``unknown``,
    - one column per annotator (``annotator_1`` ...), preserving the raw string labels.
    """
    records = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    rows = []
    for rec in records:
        row = {
            "index": rec.get("index"),
            "p_unsafe": float(rec["p_unsafe"]),
            "label": rec["label"],
            "y_true": 0 if rec["label"] == SAFE else 1,
            "risk_category": risk_category(rec["label"]),
        }
        for name, value in (rec.get("annotator_labels") or {}).items():
            row[name] = value
        rows.append(row)

    return pd.DataFrame(rows)


def auroc(y_true, p_unsafe) -> float:
    """Binary AUROC. Returns NaN (never raises) when a class is absent — e.g. the
    all-safe testset slice with zero positives — so callers get a defined value."""
    y_true = np.asarray(y_true)
    p_unsafe = np.asarray(p_unsafe, dtype=float)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p_unsafe))


def roc_sweep(y_true, p_unsafe):
    """ROC sweep as ``(fpr, tpr, thresholds)`` from ``sklearn.roc_curve``.

    ``fpr``/``tpr`` are non-decreasing and run from the (0, 0) corner to (1, 1);
    ``thresholds`` is decreasing with ``+inf`` first (the "predict nothing" point).
    """
    return roc_curve(np.asarray(y_true), np.asarray(p_unsafe, dtype=float))


def fpr_at_tpr(y_true, p_unsafe, target_tpr: float) -> dict:
    """FPR at the operating point that first reaches ``target_tpr``.

    Walks the ROC sweep (threshold high -> low, TPR non-decreasing) and takes the
    **smallest achieved TPR that is >= target** — i.e. the largest threshold still
    catching the target fraction of positives, which is the minimum-FPR point meeting
    the recall target (lower FPR is better, so this is the figure worth reporting).

    Returns ``{"fpr", "threshold", "achieved_tpr"}``. All three are NaN when the target
    is unreachable: no positives, no negatives, or ``target_tpr`` above the max TPR.
    """
    y_true = np.asarray(y_true)
    p_unsafe = np.asarray(p_unsafe, dtype=float)
    nan = {"fpr": float("nan"), "threshold": float("nan"), "achieved_tpr": float("nan")}
    if len(np.unique(y_true)) < 2:
        return nan

    # drop_intermediate=False keeps every threshold, so the exact operating point that
    # first meets the target TPR is present (the default hull-pruning can drop it).
    fpr, tpr, thresholds = roc_curve(y_true, p_unsafe, drop_intermediate=False)
    # tpr is sorted non-decreasing; first index whose tpr >= target.
    idx = int(np.searchsorted(tpr, target_tpr, side="left"))
    if idx >= len(tpr):
        return nan
    return {
        "fpr": float(fpr[idx]),
        "threshold": float(thresholds[idx]),
        "achieved_tpr": float(tpr[idx]),
    }


def fpr_at_90_95(y_true, p_unsafe) -> dict:
    """Convenience: FPR@90%TPR and FPR@95%TPR as ``{"fpr_at_90", "fpr_at_95"}``."""
    return {
        "fpr_at_90": fpr_at_tpr(y_true, p_unsafe, 0.90),
        "fpr_at_95": fpr_at_tpr(y_true, p_unsafe, 0.95),
    }


def subgroup_auroc(risk_categories, p_unsafe, subgroup: str) -> float:
    """AUROC for one unsafe subgroup's positives against all safe negatives.

    Restricts to rows whose ``risk_category`` is either ``subgroup`` or ``safe``, labels
    the subgroup rows positive, and scores against ``p_unsafe``. NaN when the subgroup
    has no positives (nothing to rank).
    """
    risk_categories = np.asarray(risk_categories)
    p_unsafe = np.asarray(p_unsafe, dtype=float)
    mask = (risk_categories == subgroup) | (risk_categories == SAFE)
    if not mask.any():
        return float("nan")
    y = (risk_categories[mask] == subgroup).astype(int)
    return auroc(y, p_unsafe[mask])


def _annotator_matrix(annotator_labels) -> np.ndarray:
    """Normalize assorted annotator containers to an ``(n_items, n_raters)`` string array.

    Accepts a DataFrame (annotator columns), a list of per-item dicts (values taken in
    sorted-key order, the seam's ``annotator_labels``), or a 2-D array-like.
    """
    if isinstance(annotator_labels, pd.DataFrame):
        return annotator_labels.to_numpy(dtype=object).astype(str)
    seq = list(annotator_labels)
    if seq and isinstance(seq[0], dict):
        seq = [[row[k] for k in sorted(row)] for row in seq]
    return np.array(seq, dtype=object).astype(str)


def fleiss_kappa_annotators(annotator_labels) -> float:
    """Fleiss' kappa across the annotators (fixed rater count, categorical labels).

    NaN when there are fewer than 2 items or the labels admit no variation (kappa is
    undefined with a single observed category).
    """
    matrix = _annotator_matrix(annotator_labels)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return float("nan")
    if len(np.unique(matrix)) < 2:
        return float("nan")
    table, _ = aggregate_raters(matrix)
    return float(fleiss_kappa(table, method="fleiss"))


def cohen_kappa(labels_a, labels_b) -> float:
    """Cohen's kappa between two raters' label sequences (thin ``sklearn`` wrapper)."""
    return float(cohen_kappa_score(list(labels_a), list(labels_b)))


def _pairwise_cohen(matrix: np.ndarray) -> dict:
    """Cohen's kappa for every rater pair, keyed ``"i_vs_j"`` (0-based columns)."""
    n_raters = matrix.shape[1]
    out = {}
    for i in range(n_raters):
        for j in range(i + 1, n_raters):
            out[f"{i}_vs_{j}"] = cohen_kappa(matrix[:, i], matrix[:, j])
    return out


def evaluate(scores) -> dict:
    """Bundle the headline report from a scores frame (or a JSONL path).

    Accepts either a DataFrame from :func:`load_scores` or a path to the seam JSONL.
    Returns AUROC, FPR@{90,95}%TPR, per-subgroup AUROC, positive/negative counts, and
    inter-annotator agreement (Fleiss' kappa plus pairwise Cohen's kappa — the human
    ceiling this classifier is chasing).
    """
    df = scores if isinstance(scores, pd.DataFrame) else load_scores(scores)

    y_true = df["y_true"].to_numpy()
    p_unsafe = df["p_unsafe"].to_numpy(dtype=float)
    risk = df["risk_category"].to_numpy()

    at = fpr_at_90_95(y_true, p_unsafe)
    report = {
        "n": int(len(df)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "auroc": auroc(y_true, p_unsafe),
        "fpr_at_90": at["fpr_at_90"],
        "fpr_at_95": at["fpr_at_95"],
        "subgroup_auroc": {
            SELF_HARM: subgroup_auroc(risk, p_unsafe, SELF_HARM),
            HARM_TO_OTHERS: subgroup_auroc(risk, p_unsafe, HARM_TO_OTHERS),
        },
    }

    annotator_cols = [c for c in df.columns if str(c).startswith("annotator_")]
    if len(annotator_cols) >= 2 and len(df) >= 2:
        matrix = df[sorted(annotator_cols)].to_numpy(dtype=object).astype(str)
        report["fleiss_kappa"] = fleiss_kappa_annotators(matrix)
        report["cohen_kappa_pairwise"] = _pairwise_cohen(matrix)
    else:
        report["fleiss_kappa"] = float("nan")
        report["cohen_kappa_pairwise"] = {}

    return report


def is_nan(value) -> bool:
    """True if ``value`` is a float NaN (readability helper for callers checking gates)."""
    return isinstance(value, float) and math.isnan(value)
