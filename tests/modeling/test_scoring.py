# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Tests for the scoring module: pure prompt rendering / softmax math as exact
unit assertions, plus a CPU end-to-end pass over a tiny model and fake dataset."""

import json
import math

import pytest
import torch

from mh_guard_harness.modeling import scoring

# Small in-vocab test ids; must match the tiny model / fake tokenizer in conftest.
TEST_SAFE_ID = 10
TEST_UNSAFE_ID = 11
TEST_PREFIX_IDS = (12, 13)

MESSAGES = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi there"},
    {"role": "user", "content": "bye"},
]


# --- pure: render_conversation / build_prompt -----------------------------------


def test_render_conversation_exact():
    assert scoring.render_conversation(MESSAGES) == "User: hello\nAssistant: hi there\nUser: bye"


def test_render_conversation_single_user_turn():
    assert scoring.render_conversation([{"role": "user", "content": "hi"}]) == "User: hi"


def test_render_conversation_unknown_role_falls_back_to_capitalize():
    msgs = [{"role": "system", "content": "x"}]
    assert scoring.render_conversation(msgs) == "System: x"


def test_build_prompt_exact(guard_prompt_path, guard_fixture_text):
    """The rendered prompt is the loaded file with the conversation substituted in —
    asserted against the fixture's own text, not against any shipped constant."""
    expected = guard_fixture_text.replace(
        "{conversation}", "User: hello\nAssistant: hi there\nUser: bye"
    )
    assert scoring.build_prompt(MESSAGES, prompt_path=guard_prompt_path) == expected


def test_build_prompt_embeds_conversation_between_markers(guard_prompt_path):
    """The conversation lands between the prompt's own delimiters, whatever they are:
    here, the fixture's."""
    prompt = scoring.build_prompt(
        [{"role": "user", "content": "need help"}], prompt_path=guard_prompt_path
    )
    assert "<<CONVO>>\nUser: need help\n<</CONVO>>" in prompt


def test_build_prompt_requires_a_path():
    from mh_guard_harness.paper_prompts import PromptPathError

    with pytest.raises(PromptPathError):
        scoring.build_prompt(MESSAGES, prompt_path=None)


# --- pure: p_unsafe_from_logits -------------------------------------------------


def test_p_unsafe_equal_logits_is_half():
    row = torch.zeros(20)
    assert scoring.p_unsafe_from_logits(row, safe_id=2, unsafe_id=5) == pytest.approx(0.5)


def test_p_unsafe_high_unsafe_logit_near_one():
    row = torch.zeros(20)
    row[5] = 100.0
    assert scoring.p_unsafe_from_logits(row, safe_id=2, unsafe_id=5) == pytest.approx(1.0)


def test_p_unsafe_high_safe_logit_near_zero():
    row = torch.zeros(20)
    row[2] = 100.0
    assert scoring.p_unsafe_from_logits(row, safe_id=2, unsafe_id=5) == pytest.approx(0.0)


def test_p_unsafe_known_softmax_value():
    # logits (safe=1.0, unsafe=2.0) -> P(unsafe) = 1/(1+e^-1).
    row = torch.zeros(20)
    row[2], row[5] = 1.0, 2.0
    expected = 1.0 / (1.0 + math.exp(-1.0))
    assert scoring.p_unsafe_from_logits(row, safe_id=2, unsafe_id=5) == pytest.approx(expected)


def test_p_unsafe_reads_only_the_two_target_ids():
    # A huge logit at an unrelated id must not affect the 2-way score.
    row = torch.zeros(20)
    row[2], row[5] = 0.0, 0.0
    row[9] = 1000.0
    assert scoring.p_unsafe_from_logits(row, safe_id=2, unsafe_id=5) == pytest.approx(0.5)


def test_default_token_ids_match_mindguard_values():
    assert scoring.SAFE_ID == 22291
    assert scoring.UNSAFE_ID == 73067
    assert scoring.PREFIX_IDS == (73037, 25)


# --- CPU integration: score_conversation / score_testset ------------------------


def test_score_conversation_returns_probability(tiny_model, fake_tokenizer, guard_prompt_path):
    p = scoring.score_conversation(
        tiny_model,
        fake_tokenizer,
        MESSAGES,
        guard_prompt_path=guard_prompt_path,
        safe_id=TEST_SAFE_ID,
        unsafe_id=TEST_UNSAFE_ID,
        prefix_ids=TEST_PREFIX_IDS,
    )
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_attach_adapter_scores_a_fine_tuned_model(
    tiny_model_factory, fake_tokenizer, tmp_path, guard_prompt_path
):
    """A trained LoRA adapter loads onto the base and scores end-to-end (the adapter-eval path),
    exercised on CPU with a tiny adapter so no GPU or 8B model is needed."""
    from peft import LoraConfig, get_peft_model

    # make + save a tiny LoRA adapter on one base instance...
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                      task_type="CAUSAL_LM")
    adapter_dir = tmp_path / "adapter"
    get_peft_model(tiny_model_factory(), lora).save_pretrained(str(adapter_dir))

    # ...then attach it to a FRESH base (as the eval script does) and score.
    wrapped = scoring.attach_adapter(tiny_model_factory(), str(adapter_dir))
    p = scoring.score_conversation(
        wrapped,
        fake_tokenizer,
        MESSAGES,
        guard_prompt_path=guard_prompt_path,
        safe_id=TEST_SAFE_ID,
        unsafe_id=TEST_UNSAFE_ID,
        prefix_ids=TEST_PREFIX_IDS,
    )
    assert isinstance(p, float) and 0.0 <= p <= 1.0


def _fake_rows():
    return [
        {
            "prompt": [{"role": "user", "content": "I feel fine today."}],
            "label": "safe",
            "annotator_labels": {
                "annotator_1": "safe",
                "annotator_2": "safe",
                "annotator_3": "safe",
            },
        },
        {
            "prompt": [
                {"role": "user", "content": "I've been struggling."},
                {"role": "assistant", "content": "I'm here."},
                {"role": "user", "content": "I want to end it tonight."},
            ],
            "label": "unsafe_self_harm_risk",
            "annotator_labels": {
                "annotator_1": "unsafe_self_harm_risk",
                "annotator_2": "unsafe_self_harm_risk",
                "annotator_3": "safe",
            },
        },
    ]


def test_score_testset_writes_seam_contract(tiny_model, fake_tokenizer, tmp_path, guard_prompt_path):
    rows = _fake_rows()
    out = tmp_path / "scores.jsonl"
    returned = scoring.score_testset(
        tiny_model,
        fake_tokenizer,
        rows,
        output_path=out,
        guard_prompt_path=guard_prompt_path,
        safe_id=TEST_SAFE_ID,
        unsafe_id=TEST_UNSAFE_ID,
        prefix_ids=TEST_PREFIX_IDS,
    )
    assert returned == out

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == len(rows)
    for i, (rec, row) in enumerate(zip(records, rows)):
        assert set(rec) == {"index", "p_unsafe", "label", "annotator_labels"}
        assert rec["index"] == i
        assert isinstance(rec["p_unsafe"], float)
        assert 0.0 <= rec["p_unsafe"] <= 1.0
        # label + annotator_labels carried straight through, uninterpreted.
        assert rec["label"] == row["label"]
        assert rec["annotator_labels"] == row["annotator_labels"]


def test_score_testset_creates_parent_dirs(tiny_model, fake_tokenizer, tmp_path, guard_prompt_path):
    out = tmp_path / "nested" / "eval" / "scores.jsonl"
    scoring.score_testset(
        tiny_model,
        fake_tokenizer,
        _fake_rows(),
        output_path=out,
        guard_prompt_path=guard_prompt_path,
        safe_id=TEST_SAFE_ID,
        unsafe_id=TEST_UNSAFE_ID,
        prefix_ids=TEST_PREFIX_IDS,
    )
    assert out.exists()


def test_score_testset_respects_limit(tiny_model, fake_tokenizer, tmp_path, guard_prompt_path):
    out = tmp_path / "scores.jsonl"
    scoring.score_testset(
        tiny_model,
        fake_tokenizer,
        _fake_rows(),
        output_path=out,
        guard_prompt_path=guard_prompt_path,
        safe_id=TEST_SAFE_ID,
        unsafe_id=TEST_UNSAFE_ID,
        prefix_ids=TEST_PREFIX_IDS,
        limit=1,
    )
    records = out.read_text().splitlines()
    assert len(records) == 1
