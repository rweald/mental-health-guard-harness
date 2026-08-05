# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Reconstruction of per-turn testset rows into aligned conversations."""

from mh_guard_harness.conversations import Conversation, reconstruct


def _row(contents, label, ann=None):
    """Build a row whose prompt alternates user/assistant and ends on a user turn.
    ``contents`` is the list of user-turn texts; assistant filler is interleaved."""
    prompt = []
    for i, c in enumerate(contents):
        if i:
            prompt.append({"role": "assistant", "content": f"reply{i}"})
        prompt.append({"role": "user", "content": c})
    return {"prompt": prompt, "label": label, "annotator_labels": ann or {}}


def test_nested_prefixes_collapse_to_one_conversation():
    rows = [
        _row(["a"], "safe"),
        _row(["a", "b"], "safe"),
        _row(["a", "b", "c"], "unsafe_self_harm_risk"),
    ]
    convs = reconstruct(rows)
    assert len(convs) == 1
    c = convs[0]
    assert c.num_user_turns == 3
    assert [g.user_turn_index for g in c.gold_turns] == [1, 2, 3]
    assert [g.label for g in c.gold_turns][-1] == "unsafe_self_harm_risk"


def test_two_independent_conversations():
    rows = [_row(["a", "b"], "safe"), _row(["x", "y"], "safe")]
    convs = reconstruct(rows)
    assert len(convs) == 2
    assert all(isinstance(c, Conversation) for c in convs)


def test_partial_gold_coverage_keeps_only_labeled_turns():
    # Only turns 1 and 3 carry gold; turn 2 is unlabeled context.
    rows = [_row(["a"], "safe"), _row(["a", "b", "c"], "unsafe_threats_to_others")]
    convs = reconstruct(rows)
    assert len(convs) == 1
    idxs = [g.user_turn_index for g in convs[0].gold_turns]
    assert idxs == [1, 3]  # turn 2 has no row -> not in gold_turns
    assert convs[0].num_user_turns == 3  # but the full context is retained


def test_shared_opener_attaches_to_both_branches():
    # "hello" opens two conversations that then diverge.
    rows = [
        _row(["hello"], "safe"),
        _row(["hello", "b"], "safe"),
        _row(["hello", "z"], "unsafe_self_harm_risk"),
    ]
    convs = reconstruct(rows)
    assert len(convs) == 2  # two maximal branches
    # the shared opener's gold turn appears in both reconstructed branches
    openers = [g for c in convs for g in c.gold_turns if g.user_turn_index == 1]
    assert len(openers) == 2
    assert all(g.label == "safe" for g in openers)


def test_msg_index_points_at_the_user_turn():
    rows = [_row(["a", "b"], "safe")]
    c = reconstruct(rows)[0]
    for g in c.gold_turns:
        assert c.messages[g.msg_index]["role"] == "user"


def test_annotator_labels_carried_through():
    ann = {"annotator_1": "safe", "annotator_2": "safe", "annotator_3": "unsafe_self_harm_risk"}
    rows = [_row(["a"], "safe", ann)]
    c = reconstruct(rows)[0]
    assert c.gold_turns[0].annotator_labels == ann
