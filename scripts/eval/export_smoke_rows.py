# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Build a small smoke set for decision-model scoring from synthetic dialogues, so the
smoke run never touches the testset. Local, torch-free.

Each labeled user turn becomes one row in the testset's shape: the past-only message
prefix ending at that turn, with the judge's label standing in for gold. The smoke
checks mechanics only (response shape, refusals, context fit, probability spread);
its numbers are not results and must not be used to change the spec.

Picks whole dialogues round-robin across the scenario risk categories (safe, self-harm,
harm to others), so the smoke exercises every option of the three-way question.

    uv run python scripts/eval/export_smoke_rows.py --dialogues-n 6
"""

import argparse
import json
from pathlib import Path

from mh_guard_harness.generation import load_dialogues
from mh_guard_harness.io_utils import open_shared
from mh_guard_harness.scenarios import RISK_CATEGORIES

DEFAULT_OUTPUT = Path("results/eval/decisions/smoke_rows.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", default="results/generated/dialogues.jsonl")
    ap.add_argument("--labels", default="results/generated/labels/glm-5.2__full.jsonl")
    ap.add_argument("--dialogues-n", type=int, default=6)
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    labels = {}
    with open(args.labels) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                labels[(r["dialogue_id"], r["user_turn_index"])] = r["judge_label"]

    dialogues = load_dialogues(args.dialogues)
    # Scenario ids are ``<risk_category>__<subcategory>__<n>``.
    by_category = [
        [d for d in dialogues if d.scenario_id.split("__")[0] == cat]
        for cat in RISK_CATEGORIES
    ]
    picked = [d for group in zip(*by_category) for d in group][: args.dialogues_n]

    n = 0
    with open_shared(args.output) as f:
        for d in picked:
            seen_users = 0
            for i, m in enumerate(d.messages):
                if m["role"] != "user":
                    continue
                seen_users += 1
                label = labels.get((d.key(), seen_users))
                if label is None:
                    continue
                f.write(json.dumps({
                    "index": n,
                    "prompt": d.messages[: i + 1],
                    "label": label,
                    "annotator_labels": {},
                    "source": {"dialogue_id": d.key(), "user_turn_index": seen_users},
                }) + "\n")
                n += 1
    print(f"wrote {n} rows from {len(picked)} dialogues -> {args.output}")


if __name__ == "__main__":
    main()
