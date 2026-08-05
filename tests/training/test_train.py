# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune wiring: config mapping validated against the container's trl/peft, on CPU.

The heavy ``run_finetune`` (model + GPU) is not unit-tested; these lock the torch-light
seams it is composed from so the real run can't silently mis-map a hyperparameter."""

import json

from mh_guard_harness.training.config import RunSpec, TrainConfig
from mh_guard_harness.training.dataset import TrainExample
from mh_guard_harness.training.train import (
    adapter_dir,
    make_lora_config,
    make_sft_config,
    to_dataset,
    write_snapshot,
)


def _examples(n=3):
    return [
        TrainExample(
            dialogue_id="d#seed0",
            user_turn_index=i + 1,
            label="safe" if i else "unsafe_self_harm_risk",
            prompt=f"prompt {i}",
            completion="Safety: Safe\nCategories: None",
        )
        for i in range(n)
    ]


def test_adapter_dir_keyed_by_judge_and_seed():
    assert adapter_dir("glm-5.2", 1) == "results/train/adapters/glm-5.2__seed1"
    assert adapter_dir("glm-4.6", 2, root="x") == "x/glm-4.6__seed2"


def test_to_dataset_prompt_completion_columns():
    ds = to_dataset(_examples(3))
    assert ds.column_names == ["prompt", "completion"]
    assert len(ds) == 3
    assert ds[0]["completion"] == "Safety: Safe\nCategories: None"


def test_lora_config_maps_frozen_fields():
    cfg = TrainConfig()
    lc = make_lora_config(cfg)
    assert lc.r == cfg.lora_r
    assert lc.lora_alpha == cfg.lora_alpha
    assert lc.lora_dropout == cfg.lora_dropout
    # peft stores target_modules as an (unordered) set
    assert set(lc.target_modules) == set(cfg.lora_target_modules)
    assert lc.task_type == "CAUSAL_LM"
    assert lc.bias == "none"


def test_sft_config_maps_batch_schedule_and_seed():
    # bf16/checkpointing off so SFTConfig constructs cleanly on a CPU-only test box;
    # the mapping wiring is what we assert, not GPU-specific validation.
    cfg = TrainConfig(bf16=False, gradient_checkpointing=False)
    run = RunSpec(judge="glm-5.2", seed=2, label_path="l.jsonl", out_dir="out/x", config=cfg)
    sft = make_sft_config(cfg, run)

    assert sft.per_device_train_batch_size == cfg.micro_batch_size
    assert sft.gradient_accumulation_steps == cfg.grad_accum_steps
    # global batch is the product — the paper's 128
    assert sft.per_device_train_batch_size * sft.gradient_accumulation_steps == 128
    assert sft.max_length == cfg.max_seq_len
    assert sft.num_train_epochs == cfg.num_train_epochs
    assert sft.learning_rate == cfg.learning_rate
    assert sft.lr_scheduler_type == cfg.lr_scheduler_type
    assert sft.warmup_ratio == cfg.warmup_ratio
    assert sft.max_grad_norm == cfg.max_grad_norm
    assert sft.packing is False
    assert sft.completion_only_loss is True
    assert sft.output_dir == "out/x"
    # seed threads to BOTH the run seed and the data shuffle, so seed N is identical
    # across the three label sets
    assert sft.seed == 2 and sft.data_seed == 2


def test_max_steps_caps_run_and_defaults_to_full():
    cfg = TrainConfig(bf16=False, gradient_checkpointing=False)
    run = RunSpec(judge="j", seed=0, label_path="l", out_dir="o", config=cfg)
    # default: -1 means "use the epoch count", not a step cap
    assert make_sft_config(cfg, run).max_steps == -1
    assert make_sft_config(cfg, run, max_steps=5).max_steps == 5


def test_sft_config_passes_bf16_flag_through():
    cfg = TrainConfig(bf16=False, gradient_checkpointing=False)
    run = RunSpec(judge="j", seed=0, label_path="l", out_dir="o", config=cfg)
    assert make_sft_config(cfg, run).bf16 is False


def test_sft_config_passes_assistant_only_loss_through():
    # the frozen field is fingerprinted, so it has to actually drive the trainer
    cfg = TrainConfig(bf16=False, gradient_checkpointing=False, assistant_only_loss=False)
    run = RunSpec(judge="j", seed=0, label_path="l", out_dir="o", config=cfg)
    assert make_sft_config(cfg, run).completion_only_loss is False


def test_write_snapshot_records_provenance_and_fingerprint(tmp_path):
    run = RunSpec(judge="glm-5.2", seed=1, label_path="l.jsonl", out_dir=str(tmp_path))
    examples = _examples(3)
    path = write_snapshot(run, examples, tmp_path)
    snap = json.loads(path.read_text())

    assert snap["judge"] == "glm-5.2" and snap["seed"] == 1
    assert snap["config_fingerprint"] == TrainConfig().fingerprint()
    assert snap["n_examples"] == 3
    assert snap["label_distribution"] == {"unsafe_self_harm_risk": 1, "safe": 2}
    assert snap["base_model"] == TrainConfig().base_model
    # the full frozen config is embedded for auditability
    assert snap["config"]["num_train_epochs"] == 3


def test_write_snapshot_records_the_guard_prompt_that_rendered_the_examples(
    tmp_path, guard_prompt_path
):
    """The config fingerprint proves the runs shared hyperparameters; this record is
    what proves they shared a prompt. Without it an adapter is not reproducible."""
    from mh_guard_harness import paper_prompts

    loaded = paper_prompts.load("guard", path=guard_prompt_path)
    run = RunSpec(
        judge="glm-5.2", seed=1, label_path="l.jsonl", out_dir=str(tmp_path),
        guard_prompt_path=str(guard_prompt_path),
    )
    snap = json.loads(
        write_snapshot(run, _examples(1), tmp_path, guard_prompt=loaded).read_text()
    )

    assert snap["guard_prompt"]["sha256"] == loaded.sha256
    assert snap["guard_prompt"]["mode"] == paper_prompts.MODE_CUSTOM
    assert snap["guard_prompt"]["hash_enforced"] is False
    # provenance travels with the adapter — no local directory paths in it
    assert str(tmp_path) not in json.dumps(snap["guard_prompt"])


def test_guard_prompt_path_is_not_part_of_the_config_fingerprint():
    """The fingerprint is a claim about hyperparameters. Two machines running the same
    sweep from different prompt paths must still fingerprint identically — prompt
    identity is proven by hash in the snapshot, not by path."""
    a = RunSpec(judge="j", seed=0, label_path="l", out_dir="o", guard_prompt_path="/a/g.txt")
    b = RunSpec(judge="j", seed=0, label_path="l", out_dir="o", guard_prompt_path="/b/g.txt")
    assert a.config.fingerprint() == b.config.fingerprint()
