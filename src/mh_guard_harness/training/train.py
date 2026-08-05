# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Run one LoRA fine-tune: one label set × one seed.

Assembles the pieces the rest of the training package defines — ``build_examples``
for the data, ``TrainConfig`` for the frozen hyperparameters — into a trl ``SFTTrainer``
run over the frozen Qwen3Guard base, and writes the adapter plus a provenance snapshot
into the training lineage. The judge's label set is the only thing that varies between
runs; everything else resolves from the shared config.

Structured so the wiring is unit-testable on CPU without the 8B model: ``to_dataset``,
``make_lora_config``, ``make_sft_config``, and ``write_snapshot`` are pure/torch-light
and validated against the same trl/peft versions the container runs. Only
``run_finetune`` touches the real model and a GPU — that step runs in the container,
behind the project's GPU sign-off gate.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from datasets import Dataset

from .. import paper_prompts
from ..generation import load_dialogues
from ..label_audit import load_label_rows
from .config import RunSpec, TrainConfig
from .dataset import TrainExample, build_examples

DEFAULT_DIALOGUES = "results/generated/dialogues.jsonl"
ADAPTER_ROOT = "results/train/adapters"


def adapter_dir(judge: str, seed: int, root: str = ADAPTER_ROOT) -> str:
    """Training-lineage output path for one run, keyed by judge + seed."""
    return f"{root}/{judge}__seed{seed}"


def to_dataset(examples: list[TrainExample]) -> Dataset:
    """A prompt/completion dataset — trl masks the prompt so the loss falls only on the
    target tokens (the paper's assistant-only loss), no chat-template generation tags
    required."""
    return Dataset.from_dict(
        {
            "prompt": [e.prompt for e in examples],
            "completion": [e.completion for e in examples],
        }
    )


def make_lora_config(cfg: TrainConfig):
    """Map the frozen LoRA fields onto a peft ``LoraConfig``."""
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def make_sft_config(cfg: TrainConfig, run: RunSpec, max_steps: int | None = None):
    """Map the frozen config + per-run bits onto a trl ``SFTConfig``. Global batch is
    reached as ``micro_batch_size`` × ``grad_accum_steps`` on the single GPU.

    ``max_steps`` caps the run at a fixed number of optimizer steps (overriding the
    epoch count) — used for the smoke run and debugging, left ``None`` for real runs."""
    from trl import SFTConfig

    return SFTConfig(
        output_dir=run.out_dir,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        adam_epsilon=cfg.adam_epsilon,
        max_grad_norm=cfg.max_grad_norm,
        max_length=cfg.max_seq_len,
        packing=cfg.packing,
        completion_only_loss=cfg.assistant_only_loss,  # mask the prompt; train on the target only
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        seed=run.seed,
        data_seed=run.seed,  # same shuffle for seed N across all three label sets
        logging_steps=1,
        save_strategy="no",  # one save at the end via save_model
        report_to=[],
    )


def write_snapshot(
    run: RunSpec,
    examples: list[TrainExample],
    out_dir: str | Path,
    *,
    guard_prompt: paper_prompts.LoadedPrompt | None = None,
) -> Path:
    """Record what produced this adapter: the run's variables, the full frozen config,
    its fingerprint, the realized example count/label mix, and which guard prompt the
    examples were rendered with. The fingerprint lets the adapters be proven to share
    identical hyperparameters; the prompt hash lets them be proven to share a prompt."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "judge": run.judge,
        "seed": run.seed,
        "label_path": run.label_path,
        "base_model": run.config.base_model,
        "config_fingerprint": run.config.fingerprint(),
        "config": run.config.to_dict(),
        "n_examples": len(examples),
        "label_distribution": dict(Counter(e.label for e in examples)),
        "guard_prompt": paper_prompts.provenance(guard_prompt) if guard_prompt else None,
    }
    path = out / "run.json"
    path.write_text(json.dumps(snapshot, indent=2))
    return path


def run_finetune(
    run: RunSpec,
    dialogues_path: str = DEFAULT_DIALOGUES,
    max_steps: int | None = None,
) -> Path:
    """Train one LoRA adapter and save it + its provenance snapshot. Loads the 8B base
    onto the GPU — container-only, behind the GPU sign-off gate. ``max_steps`` caps the
    run for a smoke test; leave ``None`` for a real run."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTTrainer

    cfg = run.config
    set_seed(run.seed)

    # Load the guard prompt once up front: it fails loudly before the 8B base is pulled
    # onto the GPU, and the LoadedPrompt goes straight into the provenance snapshot.
    guard_prompt = paper_prompts.load(
        "guard",
        path=run.guard_prompt_path,
        mode=paper_prompts.mode_for(run.reproduction_mode),
    )

    dialogues = load_dialogues(dialogues_path)
    labels = load_label_rows(run.label_path)
    examples = build_examples(
        dialogues,
        labels,
        guard_prompt_path=run.guard_prompt_path,
        reproduction_mode=run.reproduction_mode,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        attn_implementation=cfg.attn_implementation,
    )
    if cfg.gradient_checkpointing:
        model.config.use_cache = False  # incompatible with checkpointing

    trainer = SFTTrainer(
        model=model,
        args=make_sft_config(cfg, run, max_steps=max_steps),
        train_dataset=to_dataset(examples),
        peft_config=make_lora_config(cfg),
        processing_class=tokenizer,
    )
    trainer.train()

    out = Path(run.out_dir)
    trainer.save_model(str(out))
    write_snapshot(run, examples, out, guard_prompt=guard_prompt)
    (out / "train_log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in trainer.state.log_history)
    )
    return out
