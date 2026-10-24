# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Interim spec-tuning scripts (scripts/spec_tuning/): synthetic row building. Loaded
from the script directory, since that code is deliberately not part of the package."""

import importlib.util
import json
from pathlib import Path

from mh_guard_harness.generation import Dialogue

_PATH = Path(__file__).parents[1] / "scripts" / "spec_tuning" / "dev_rows.py"
_spec = importlib.util.spec_from_file_location("spec_tuning_dev_rows", _PATH)
dev_rows = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dev_rows)


def test_rows_from_dialogues_one_row_per_judged_user_turn(tmp_path):
    d = Dialogue(scenario_id="harm_to_others__direct_threat__000", seed=0, messages=[
        {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ])
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text("\n".join(json.dumps(r) for r in [
        {"dialogue_id": d.key(), "user_turn_index": 1, "judge_label": "safe"},
        {"dialogue_id": d.key(), "user_turn_index": 3,
         "judge_label": "unsafe_threats_to_others"},
    ]) + "\n")
    rows = dev_rows.rows_from_dialogues([d], dev_rows.load_judge_labels(labels_path))
    assert [r["index"] for r in rows] == [0, 1]
    assert [r["prompt"][-1]["content"] for r in rows] == ["u1", "u3"]  # turn 2 unjudged
    assert len(rows[1]["prompt"]) == 5  # past-only prefix ends at the judged turn
    assert rows[1]["source"] == {"dialogue_id": d.key(), "user_turn_index": 3}
    assert rows[1]["label"] == "unsafe_threats_to_others"
