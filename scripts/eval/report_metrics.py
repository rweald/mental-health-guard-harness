# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Local / torch-free: compute the validation metrics from the scores JSONL.

Reads the seam-contract file written by score_testset.py and prints AUROC, FPR@90/95%
TPR, per-subgroup AUROC, and inter-annotator agreement. Runs in the local uv env (which
has scikit-learn / scipy / statsmodels); the GPU container does not carry those.

    uv run python scripts/eval/report_metrics.py
    uv run python scripts/eval/report_metrics.py --scores results/eval/mindguard8b_scores.jsonl
"""

import argparse
import json

from mh_guard_harness import metrics

# Kept in sync with scoring.DEFAULT_OUTPUT by hand rather than importing that module,
# so this reporter stays torch-free and doesn't drag in transformers.
DEFAULT_SCORES = "results/eval/mindguard8b_scores.jsonl"
PAPER_AUROC = 0.982


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=DEFAULT_SCORES)
    args = ap.parse_args()

    report = metrics.evaluate(args.scores)
    print(json.dumps(report, indent=2, default=str))

    auroc = report.get("auroc")
    if auroc is not None and auroc == auroc:  # not NaN
        delta = auroc - PAPER_AUROC
        print(f"\nAUROC {auroc:.4f}  vs paper ~{PAPER_AUROC}  (Δ {delta:+.4f})")


if __name__ == "__main__":
    main()
