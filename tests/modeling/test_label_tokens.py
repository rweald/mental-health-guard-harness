# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""derive_label_tokens: find a tokenizer's decision tokens by target divergence."""

import pytest

from mh_guard_harness.modeling.scoring import derive_label_tokens, LabelTokens


class FakeTokenizer:
    """Encodes only the exact strings in its table — enough to drive the divergence
    logic without a real BPE model."""

    def __init__(self, table):
        self.table = table

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(self.table[text])


def test_single_token_labels():
    tok = FakeTokenizer({"Safety: Safe": [10, 11, 20], "Safety: Unsafe": [10, 11, 21]})
    lt = derive_label_tokens(tok)
    assert lt == LabelTokens(safe_id=20, unsafe_id=21, prefix_ids=(10, 11))


def test_multi_token_unsafe_uses_first_divergent_token():
    # " Unsafe" splits into two pieces; scoring only needs the first differing token.
    tok = FakeTokenizer({"Safety: Safe": [10, 11, 20], "Safety: Unsafe": [10, 11, 30, 31]})
    lt = derive_label_tokens(tok)
    assert lt.safe_id == 20 and lt.unsafe_id == 30
    assert lt.prefix_ids == (10, 11)


def test_empty_prefix_when_labels_diverge_immediately():
    tok = FakeTokenizer({"Safety: Safe": [5, 99], "Safety: Unsafe": [6, 99]})
    lt = derive_label_tokens(tok)
    assert lt.prefix_ids == () and lt.safe_id == 5 and lt.unsafe_id == 6


def test_raises_when_one_target_is_a_prefix_of_the_other():
    tok = FakeTokenizer({"Safety: Safe": [10, 11], "Safety: Unsafe": [10, 11, 21]})
    with pytest.raises(ValueError, match="do not diverge"):
        derive_label_tokens(tok)


def test_custom_stem_is_encoded():
    seen = {}

    class RecordingTokenizer(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            seen[text] = True
            return super().encode(text, add_special_tokens=add_special_tokens)

    tok = RecordingTokenizer({"S: Safe": [1, 2], "S: Unsafe": [1, 3]})
    lt = derive_label_tokens(tok, stem="S:")
    assert lt.safe_id == 2 and lt.unsafe_id == 3
    assert "S: Safe" in seen and "S: Unsafe" in seen
