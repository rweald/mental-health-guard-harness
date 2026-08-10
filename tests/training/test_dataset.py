# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Example-builder: past-only context in, full-context judge label as target."""

import pytest

from mh_guard_harness.generation import Dialogue
from mh_guard_harness.training.dataset import (
    build_examples,
    render_target,
    TrainExample,
)


def _dialogue(scenario_id, seed, user_texts):
    """A user-first, alternating dialogue: u1, a1, u2, a2, ... one assistant per user."""
    messages = []
    for i, u in enumerate(user_texts, start=1):
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": f"A{i}"})
    tags = ["SAFE"] * len(user_texts)
    return Dialogue(scenario_id=scenario_id, seed=seed, messages=messages, turn_tags=tags)


def _rows(dialogue_id, labels):
    return [
        {"dialogue_id": dialogue_id, "user_turn_index": i + 1, "judge_label": lab}
        for i, lab in enumerate(labels)
    ]


# ── render_target ────────────────────────────────────────────────────────────
def test_render_target_maps_each_label():
    assert render_target("safe") == "Safety: Safe\nCategories: None"
    assert render_target("unsafe_self_harm_risk") == "Safety: Unsafe\nCategories: S1"
    assert render_target("unsafe_threats_to_others") == "Safety: Unsafe\nCategories: S2"


def test_render_target_rejects_unknown_label():
    with pytest.raises(ValueError, match="unknown judge label"):
        render_target("unsafe_other")


# ── build_examples ───────────────────────────────────────────────────────────
def test_one_example_per_labeled_turn(guard_prompt_path):
    dl = _dialogue("s", 0, ["U1", "U2", "U3"])
    rows = _rows("s#seed0", ["safe", "unsafe_self_harm_risk", "safe"])
    examples = build_examples([dl], rows, guard_prompt_path=guard_prompt_path)
    assert len(examples) == 3
    assert [e.user_turn_index for e in examples] == [1, 2, 3]
    assert [e.label for e in examples] == ["safe", "unsafe_self_harm_risk", "safe"]
    assert all(isinstance(e, TrainExample) for e in examples)


def test_prompt_is_past_only_never_leaks_future_turns(guard_prompt_path):
    # Distinctive contents so we can prove which turns are visible.
    dl = _dialogue("s", 0, ["UONE", "UTWO", "UTHREE"])
    rows = _rows("s#seed0", ["safe", "safe", "safe"])
    turn2 = build_examples([dl], rows, guard_prompt_path=guard_prompt_path)[1]  # the example labeling user turn 2

    # Past + current turn are present...
    assert "UONE" in turn2.prompt
    assert "UTWO" in turn2.prompt
    assert "A1" in turn2.prompt  # the clinician reply that preceded turn 2
    # ...but nothing from the future may leak in.
    assert "UTHREE" not in turn2.prompt
    assert "A2" not in turn2.prompt  # turn 2's own clinician reply is future context


def test_prompt_ends_at_the_labeled_user_turn(guard_prompt_path):
    dl = _dialogue("s", 0, ["UONE", "UTWO", "UTHREE"])
    rows = _rows("s#seed0", ["safe", "safe", "safe"])
    turn2 = build_examples([dl], rows, guard_prompt_path=guard_prompt_path)[1]
    # The rendered conversation block's final user line is the turn being labeled.
    convo = turn2.prompt.split("<<CONVO>>")[1].split("<</CONVO>>")[0]
    assert convo.strip().splitlines()[-1] == "User: UTWO"


def test_completion_matches_the_judge_label(guard_prompt_path):
    dl = _dialogue("s", 0, ["U1", "U2"])
    rows = _rows("s#seed0", ["safe", "unsafe_self_harm_risk"])
    examples = build_examples([dl], rows, guard_prompt_path=guard_prompt_path)
    assert examples[0].completion == "Safety: Safe\nCategories: None"
    assert examples[1].completion == "Safety: Unsafe\nCategories: S1"


def test_only_turns_in_both_sets_are_emitted(guard_prompt_path):
    dl = _dialogue("s", 0, ["U1", "U2", "U3"])
    # Label set is missing turn 2 entirely.
    rows = [
        {"dialogue_id": "s#seed0", "user_turn_index": 1, "judge_label": "safe"},
        {"dialogue_id": "s#seed0", "user_turn_index": 3, "judge_label": "safe"},
    ]
    examples = build_examples([dl], rows, guard_prompt_path=guard_prompt_path)
    assert [e.user_turn_index for e in examples] == [1, 3]


def test_duplicate_label_rows_are_rejected(guard_prompt_path):
    # last-write-wins would silently pick one of the two and break the
    # identical-example-sequence guarantee across label sets
    dl = _dialogue("s", 0, ["U1"])
    rows = _rows("s#seed0", ["safe"]) + _rows("s#seed0", ["unsafe_self_harm_risk"])
    with pytest.raises(ValueError, match="duplicate label row"):
        build_examples([dl], rows, guard_prompt_path=guard_prompt_path)


def test_deterministic_order_dialogue_then_turn(guard_prompt_path):
    a = _dialogue("a", 0, ["A1", "A2"])
    b = _dialogue("b", 0, ["B1", "B2"])
    rows = _rows("a#seed0", ["safe", "safe"]) + _rows("b#seed0", ["safe", "safe"])
    ids = [(e.dialogue_id, e.user_turn_index) for e in build_examples([a, b], rows, guard_prompt_path=guard_prompt_path)]
    assert ids == [("a#seed0", 1), ("a#seed0", 2), ("b#seed0", 1), ("b#seed0", 2)]


def test_to_chat_shape(guard_prompt_path):
    dl = _dialogue("s", 0, ["U1"])
    rows = _rows("s#seed0", ["unsafe_threats_to_others"])
    chat = build_examples([dl], rows, guard_prompt_path=guard_prompt_path)[0].to_chat()
    assert [m["role"] for m in chat] == ["user", "assistant"]
    assert chat[1]["content"] == "Safety: Unsafe\nCategories: S2"
