# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Export the testset's per-turn rows to JSONL so API scoring can run locally (container,
no GPU). The testset lives in the container's offline HF cache; decision-model scoring
is torch-free and runs in the local env, so it reads this file instead.

Row ``index`` is the dataset's own order — the same index ``score_testset.py`` writes —
so decision-model scores pair row-for-row with MindGuard-8B and the fine-tuned arms.

    ./scripts/eval/run.sh python scripts/eval/export_testset_rows.py
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from mh_guard_harness.io_utils import open_shared

DEFAULT_OUTPUT = Path("results/eval/testset_rows.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="swordhealth/MindGuard-testset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    with open_shared(args.output) as f:
        for index, r in enumerate(ds):
            f.write(json.dumps({
                "index": index,
                "prompt": r["prompt"],
                "label": r["label"],
                "annotator_labels": r["annotator_labels"],
            }) + "\n")
    print(f"wrote {len(ds)} rows -> {args.output}")


if __name__ == "__main__":
    main()
