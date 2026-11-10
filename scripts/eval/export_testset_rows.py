# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Export the testset's per-turn rows to JSONL for decision-model (API) scoring, which is
torch-free and reads this file instead of the dataset. The export needs no GPU and runs
either locally (pulls from the Hugging Face Hub; needs ``hf auth login``) or in the eval
container (reads its offline HF cache; the container itself requires a GPU).

Row ``index`` is the dataset's own order — the same index ``score_testset.py`` writes —
so decision-model scores pair row-for-row with MindGuard-8B and the fine-tuned arms.

    uv run hf auth login                                          # local
    uv run python scripts/eval/export_testset_rows.py

    ./scripts/eval/run.sh python scripts/eval/export_testset_rows.py   # container
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
