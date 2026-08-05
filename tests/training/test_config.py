# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""The frozen TrainConfig: canonical defaults, derived batch, and a stable fingerprint."""

import dataclasses

import pytest

from mh_guard_harness.training.config import RunSpec, SEEDS, TrainConfig


def test_canonical_defaults():
    c = TrainConfig()
    assert c.base_model == "Qwen/Qwen3Guard-Gen-8B"
    assert c.num_train_epochs == 3
    assert c.max_seq_len == 4096
    assert c.assistant_only_loss is True
    assert c.global_batch_size == 128


def test_grad_accum_reaches_global_batch():
    c = TrainConfig()
    assert c.grad_accum_steps == c.global_batch_size // c.micro_batch_size
    assert c.micro_batch_size * c.grad_accum_steps == 128


def test_indivisible_batch_is_rejected():
    with pytest.raises(ValueError, match="not divisible"):
        TrainConfig(global_batch_size=128, micro_batch_size=5)


def test_positive_micro_batch_enforced():
    # checked before the divisibility modulo, so 0 is a clear ValueError not a
    # ZeroDivisionError
    with pytest.raises(ValueError, match="micro_batch_size must be positive"):
        TrainConfig(micro_batch_size=0)


def test_positive_lora_rank_enforced():
    with pytest.raises(ValueError, match="lora_r must be positive"):
        TrainConfig(lora_r=0)


def test_config_is_frozen():
    c = TrainConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.learning_rate = 1e-3


def test_fingerprint_is_stable_across_instances():
    assert TrainConfig().fingerprint() == TrainConfig().fingerprint()


def test_fingerprint_changes_with_any_hyperparameter():
    base = TrainConfig().fingerprint()
    assert TrainConfig(learning_rate=1e-4).fingerprint() != base
    assert TrainConfig(lora_r=8).fingerprint() != base


def test_to_dict_is_json_safe():
    import json

    d = TrainConfig().to_dict()
    round_tripped = json.loads(json.dumps(d))  # must not raise
    # tuples serialize to JSON arrays -> lists on the way back
    assert isinstance(round_tripped["lora_target_modules"], list)
    assert len(round_tripped["lora_target_modules"]) == 7


def test_seeds_are_three_and_shared():
    assert SEEDS == (0, 1, 2)


def test_runspec_carries_only_the_variables_and_a_shared_config():
    spec = RunSpec(judge="glm-5.2", seed=1, label_path="x.jsonl", out_dir="out/")
    assert spec.judge == "glm-5.2" and spec.seed == 1
    # every run defaults to the same canonical config
    assert spec.config.fingerprint() == TrainConfig().fingerprint()
