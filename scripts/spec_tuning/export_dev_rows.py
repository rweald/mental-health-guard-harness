# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Split the synthetic dialogues into a tuning set and a holdout set for iterating on
the decision spec. Local, torch-free. The testset plays no part in spec iteration.

The split is by dialogue (one dialogue per scenario, so also by scenario): every turn
of a conversation lands on the same side, or near-duplicate context would leak between
them. It is stratified by scenario subcategory, so both sides get each kind of
conversation, and deterministic — dialogues are ordered by a hash of their id, not a
random draw — so rerunning gives the same split.

Iterate against ``tune_rows.jsonl`` only. Score ``holdout_rows.jsonl`` once, with the
final spec, to check the gains weren't fitted to the tuning dialogues.

Labels are the GLM-5.2 judge's, not clinicians', so the spec is being tuned toward the
judge's reading of each turn.

    uv run python scripts/spec_tuning/export_dev_rows.py
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from dev_rows import load_judge_labels, rows_from_dialogues

from mh_guard_harness.generation import load_dialogues
from mh_guard_harness.io_utils import open_shared

DEFAULT_OUTDIR = Path("results/spec_tuning")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", default="results/generated/dialogues.jsonl")
    ap.add_argument("--labels", default="results/generated/labels/glm-5.2__full.jsonl")
    ap.add_argument("--holdout-fraction", type=float, default=0.3)
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = ap.parse_args()

    strata = defaultdict(list)
    for d in load_dialogues(args.dialogues):
        category, subcategory = d.scenario_id.split("__")[:2]
        strata[(category, subcategory)].append(d)

    tune, holdout = [], []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda d: hashlib.sha256(d.key().encode()).hexdigest())
        n_hold = round(len(group) * args.holdout_fraction)
        holdout.extend(group[:n_hold])
        tune.extend(group[n_hold:])

    labels = load_judge_labels(args.labels)
    for name, picked in (("tune", tune), ("holdout", holdout)):
        rows = rows_from_dialogues(picked, labels)
        out = Path(args.outdir) / f"{name}_rows.jsonl"
        with open_shared(out) as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        counts = Counter(r["label"] for r in rows)
        print(f"{name:8} {len(picked):3} dialogues, {len(rows):4} rows -> {out}  {dict(counts)}")


if __name__ == "__main__":
    main()
