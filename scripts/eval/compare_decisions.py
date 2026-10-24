# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Compare the decision models against MindGuard-8B and the GLM-5.2 fine-tune. Local,
torch-free, no API calls.

Reads the scores files written by score_decisions.py, the MindGuard-8B baseline, and the
three GLM-5.2 seed files, and reports per-model metrics under both failure policies,
cost and latency, the three-way confusion, and a paired bootstrap of every model and
question against both baselines (and of the two questions against each other).

    uv run python scripts/eval/compare_decisions.py
    uv run python scripts/eval/compare_decisions.py --json > results/eval/decisions/comparison.json
"""

import argparse
import json

from mh_guard_harness import decision_report

DEFAULT_DECISIONS_DIR = "results/eval/decisions"
DEFAULT_MINDGUARD = "results/eval/mindguard8b_scores.jsonl"
DEFAULT_GLM = [f"results/eval/glm-5.2__seed{s}_scores.jsonl" for s in (0, 1, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions-dir", default=DEFAULT_DECISIONS_DIR)
    ap.add_argument("--mindguard", default=DEFAULT_MINDGUARD)
    ap.add_argument("--glm", nargs="+", default=DEFAULT_GLM)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    ap.add_argument("--workers", type=int, default=None, help="bootstrap processes")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args()

    report = decision_report.build(
        decisions_dir=args.decisions_dir,
        mindguard_path=args.mindguard,
        glm_paths=args.glm,
        n_boot=args.n_boot,
        seed=args.seed,
        workers=args.workers,
    )
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(decision_report.render(report))


if __name__ == "__main__":
    main()
