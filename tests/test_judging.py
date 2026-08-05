# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""§7.2 judge labeler: parsing, voting, and the two context modes (fake model)."""

import pytest

from mh_guard_harness.conversations import Conversation, GoldTurn
from mh_guard_harness.judging import (
    MODE_FULL,
    MODE_TURN,
    build_judge_prompt,
    label_conversation,
    majority_vote,
    parse_labels,
    render_transcript,
)
from mh_guard_harness.models import Completion


class QueueModel:
    """Returns queued texts and records each prompt it was given (for context checks)."""

    def __init__(self, texts):
        self._texts = list(texts)
        self.prompts = []

    def complete(self, messages, **opts):
        self.prompts.append(messages[0]["content"])
        return Completion(text=self._texts.pop(0))


def _conv(user_texts, gold_indices):
    msgs, gold = [], []
    for i, t in enumerate(user_texts):
        if i:
            msgs.append({"role": "assistant", "content": f"r{i}"})
        msgs.append({"role": "user", "content": t})
    for idx in gold_indices:
        msg_index = 2 * (idx - 1)  # user turns sit at even msg positions here
        gold.append(GoldTurn(user_turn_index=idx, msg_index=msg_index, label="safe"))
    return Conversation(id="c", messages=msgs, gold_turns=gold)


def test_render_transcript_numbers_from_one():
    assert render_transcript(["a", "b"]) == "USER_TURN_1: a\nUSER_TURN_2: b"


def test_render_transcript_flattens_newlines_within_a_turn():
    # a multi-line turn must stay on one line, else it misaligns the judge's numbering
    out = render_transcript(["line one\nline two", "b"])
    assert out == "USER_TURN_1: line one line two\nUSER_TURN_2: b"
    assert out.count("\n") == 1  # exactly the one separator between turns


def test_build_prompt_embeds_transcript_between_the_prompts_own_markers(judge_prompt_path):
    """The numbered transcript is substituted into whatever prompt was supplied —
    asserted against the fixture's own delimiters, not any shipped prompt's."""
    p = build_judge_prompt(["hello"], prompt_path=judge_prompt_path)
    assert "<<TURNS>>\nUSER_TURN_1: hello\n<</TURNS>>" in p


def test_build_judge_prompt_requires_a_path():
    from mh_guard_harness.paper_prompts import PromptPathError

    with pytest.raises(PromptPathError):
        build_judge_prompt(["hello"], prompt_path=None)


def test_parse_labels_basic():
    text = "USER_TURN_1: safe\nUSER_TURN_2: unsafe_self_harm_risk"
    assert parse_labels(text) == {1: "safe", 2: "unsafe_self_harm_risk"}


def test_parse_labels_normalizes_aliases_and_case():
    text = "USER_TURN_1: SAFE\nUSER_TURN_2: unsafe_harm_to_others"
    # harm_to_others spelling normalizes to the judge's canonical threats label
    assert parse_labels(text) == {1: "safe", 2: "unsafe_threats_to_others"}


def test_parse_labels_ignores_noise_and_unknown_labels():
    text = "Sure!\nUSER_TURN_1: banana\nUSER_TURN_2: safe\n"
    assert parse_labels(text) == {2: "safe"}  # unknown label dropped


def test_majority_vote_plurality():
    assert majority_vote(["safe", "safe", "unsafe_self_harm_risk"]) == "safe"


def test_majority_vote_tie_breaks_to_more_severe():
    # 2 vs 2 tie -> the more severe label wins (safety-conservative)
    v = majority_vote(["safe", "safe", "unsafe_self_harm_risk", "unsafe_self_harm_risk"])
    assert v == "unsafe_self_harm_risk"


def test_majority_vote_empty_is_none():
    assert majority_vote([]) is None


def test_full_mode_one_prompt_per_sample_sees_all_turns(judge_prompt_path):
    conv = _conv(["a", "b", "c"], gold_indices=[1, 3])
    # 3 samples, each a full labeling of all 3 turns
    labelings = [
        "USER_TURN_1: safe\nUSER_TURN_2: safe\nUSER_TURN_3: unsafe_self_harm_risk",
        "USER_TURN_1: safe\nUSER_TURN_2: safe\nUSER_TURN_3: unsafe_self_harm_risk",
        "USER_TURN_1: safe\nUSER_TURN_2: safe\nUSER_TURN_3: safe",
    ]
    model = QueueModel(labelings)
    out = label_conversation(
        model, conv, mode=MODE_FULL, n_samples=3, judge_prompt_path=judge_prompt_path
    )
    assert out == {1: "safe", 3: "unsafe_self_harm_risk"}  # vote 2:1 -> unsafe
    # full mode: exactly n_samples calls, each transcript holds all 3 user turns
    assert len(model.prompts) == 3
    for p in model.prompts:
        assert "USER_TURN_1: a" in p and "USER_TURN_2: b" in p and "USER_TURN_3: c" in p


def test_turn_mode_truncates_context_per_target_turn(judge_prompt_path):
    conv = _conv(["a", "b", "c"], gold_indices=[1, 3])
    # turn 1: 2 samples see only turn 1; turn 3: 2 samples see turns 1..3
    texts = [
        "USER_TURN_1: safe",
        "USER_TURN_1: safe",
        "USER_TURN_1: safe\nUSER_TURN_2: safe\nUSER_TURN_3: unsafe_self_harm_risk",
        "USER_TURN_1: safe\nUSER_TURN_2: safe\nUSER_TURN_3: unsafe_self_harm_risk",
    ]
    model = QueueModel(texts)
    out = label_conversation(
        model, conv, mode=MODE_TURN, n_samples=2, judge_prompt_path=judge_prompt_path
    )
    assert out == {1: "safe", 3: "unsafe_self_harm_risk"}
    # turn 1's 2 prompts see only turn 1 (no future); turn 3's see all 3
    assert len(model.prompts) == 4
    assert all("USER_TURN_2: b" not in p for p in model.prompts[:2])
    assert all("USER_TURN_1: a" in p for p in model.prompts[:2])
    assert all("USER_TURN_3: c" in p for p in model.prompts[2:])


def test_unknown_mode_raises(judge_prompt_path):
    with pytest.raises(ValueError, match="unknown mode"):
        label_conversation(
            QueueModel([]), _conv(["a"], [1]), mode="sideways",
            judge_prompt_path=judge_prompt_path,
        )
