# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Step: judge-agreement (container). Reconstruct the testset's per-turn rows into
full conversations with aligned gold turns, and dump them to JSON so the local
judge-labeling step can run without the dataset cache.

    ./scripts/eval/run.sh python scripts/eval/extract_conversations.py
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from datasets import load_dataset

from mh_guard_harness.conversations import reconstruct
from mh_guard_harness.io_utils import open_shared

DEFAULT_OUTPUT = Path("results/eval/testset_conversations.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="swordhealth/MindGuard-testset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    rows = [
        {"prompt": r["prompt"], "label": r["label"], "annotator_labels": r["annotator_labels"]}
        for r in ds
    ]
    convs = reconstruct(rows)

    out = Path(args.output)
    payload = [asdict(c) for c in convs]
    with open_shared(out) as f:
        f.write(json.dumps(payload, indent=2))

    n_gold = sum(len(c.gold_turns) for c in convs)
    print(f"{len(rows)} rows -> {len(convs)} conversations, {n_gold} gold turns")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
