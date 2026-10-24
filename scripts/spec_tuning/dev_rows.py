# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Turn judge-labeled synthetic dialogues into testset-shaped rows for spec tuning."""

import json


def load_judge_labels(path) -> dict[tuple[str, int], str]:
    """``(dialogue_id, user_turn_index) -> judge_label`` from a judge labels file."""
    labels = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                labels[(r["dialogue_id"], r["user_turn_index"])] = r["judge_label"]
    return labels


def rows_from_dialogues(dialogues, labels: dict) -> list[dict]:
    """One row per judged user turn: the past-only message prefix ending at that turn,
    with the judge's label standing in for gold. Indexes run from 0 in dialogue order;
    ``source`` keeps the way back to the dialogue and turn."""
    rows = []
    for d in dialogues:
        seen_users = 0
        for i, m in enumerate(d.messages):
            if m["role"] != "user":
                continue
            seen_users += 1
            label = labels.get((d.key(), seen_users))
            if label is None:
                continue
            rows.append({
                "index": len(rows),
                "prompt": d.messages[: i + 1],
                "label": label,
                "annotator_labels": {},
                "source": {"dialogue_id": d.key(), "user_turn_index": seen_users},
            })
    return rows
