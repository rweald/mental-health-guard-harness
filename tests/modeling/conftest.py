# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""CPU fixtures for scoring tests: a tiny random-weight causal LM and a fake
tokenizer with a small vocab, so ``score_conversation`` / ``score_testset`` run
end-to-end in seconds without MindGuard-8B or a GPU."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

VOCAB_SIZE = 64
# Small in-vocab ids standing in for the real MindGuard decision tokens.
TEST_SAFE_ID = 10
TEST_UNSAFE_ID = 11
TEST_PREFIX_IDS = (12, 13)


class FakeTokenizer:
    """Minimal stand-in for a HF tokenizer: turns the rendered prompt into a short
    deterministic id sequence (all < VOCAB_SIZE), matching the ``apply_chat_template``
    call shape ``score_conversation`` relies on."""

    def apply_chat_template(
        self, messages, add_generation_prompt=False, return_tensors=None, return_dict=False
    ):
        content = messages[-1]["content"]
        ids = [(ord(c) % (VOCAB_SIZE - 2)) + 1 for c in content[:24]] or [1]
        input_ids = torch.tensor([ids], dtype=torch.long)
        assert return_tensors == "pt" and return_dict
        return {"input_ids": input_ids}


def make_tiny_model():
    """A fresh tiny causal LM. A plain function (not just the fixture) so tests that mutate
    the model in place — e.g. wrapping it with a LoRA adapter — get their own instance."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_model():
    return make_tiny_model()


@pytest.fixture
def tiny_model_factory():
    """Build fresh tiny models on demand — for tests that mutate the model in place."""
    return make_tiny_model


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()
