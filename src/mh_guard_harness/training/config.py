# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""The single frozen training configuration shared by every fine-tune.

Internal validity hinges on ONE thing varying across the runs: the labeling judge.
So every hyperparameter that could move the classifier — base model, LoRA shape,
optimizer, schedule, batch, sequence length — lives here as one frozen dataclass,
and only ``(label_set, seed, out_dir)`` are passed alongside it per run. The three
label sets × three seeds all resolve the same ``TrainConfig``; ``fingerprint()`` is
logged with each adapter so we can later *prove* all nine shared identical settings.

Shapes follow the reference method (§6), adjusted for LoRA: the paper's
peak LR 2e-5 is for full fine-tuning, so we use the LoRA band (~1e-4–2e-4); the rest
of the schedule shape (epochs, global batch, cosine + 10% warmup, grad clip,
assistant-only loss, seq len) is held aligned. No torch import here — pure config, so
it stays CPU-importable and unit-testable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

# The three seeds every label set is trained under (report mean ± spread across them).
# Shared across arms, so seed 0 of GLM-4.6 sees the same shuffle as seed 0 of GLM-5.2.
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class TrainConfig:
    """Frozen hyperparameters shared by all nine fine-tunes. Defaults ARE the canonical
    values; instantiating with no arguments yields the config every run uses."""

    # ── base model ──
    base_model: str = "Qwen/Qwen3Guard-Gen-8B"

    # ── LoRA adapter (bf16 LoRA, not QLoRA — avoids bitsandbytes on Blackwell) ──
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # All linear projections in attention + MLP — the standard high-coverage LoRA set
    # for Qwen3-family models.
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # ── optimizer + schedule (paper §6 shape; LoRA LR band) ──
    learning_rate: float = 2e-4          # LoRA band; paper's 2e-5 is full-FT only
    num_train_epochs: int = 3
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.10           # linear warmup for 10% of steps
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # ── batch (global 128 via micro-batch × grad-accum on the single GB10) ──
    global_batch_size: int = 128
    micro_batch_size: int = 4            # paper's per-GPU micro-batch; tune at the smoke gate if OOM

    # ── sequence + loss ──
    max_seq_len: int = 4096              # paper value; our longest example is well under
    assistant_only_loss: bool = True     # mask all non-assistant tokens (paper §6)
    packing: bool = False                # off, so assistant-only masking stays per-example

    # ── precision + kernels ──
    bf16: bool = True
    gradient_checkpointing: bool = True  # memory headroom for 8B + 4096 on one GPU
    attn_implementation: str = "sdpa"    # SDPA fallback (don't depend on FA2 building on Blackwell)

    def __post_init__(self) -> None:
        if self.micro_batch_size <= 0:
            raise ValueError(
                f"micro_batch_size must be positive, got {self.micro_batch_size}"
            )
        if self.global_batch_size % self.micro_batch_size != 0:
            raise ValueError(
                f"global_batch_size {self.global_batch_size} not divisible by "
                f"micro_batch_size {self.micro_batch_size}"
            )
        if self.lora_r <= 0:
            raise ValueError(f"lora_r must be positive, got {self.lora_r}")

    @property
    def grad_accum_steps(self) -> int:
        """Accumulation steps to reach ``global_batch_size`` on a single device."""
        return self.global_batch_size // self.micro_batch_size

    def to_dict(self) -> dict:
        """Config as a plain dict (tuples become lists) — for JSON snapshots."""
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable short hash over all frozen fields. Logged with each adapter so the
        nine runs can be proven to share identical hyperparameters."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class RunSpec:
    """The only things that vary per run: which judge's labels, which seed, where the
    adapter lands. Everything else comes from the shared ``TrainConfig``.

    ``guard_prompt_path`` is per-run wiring rather than a frozen hyperparameter — the
    same prompt can live at different paths on different machines, so the path itself
    proves nothing. What must be identical across runs is the prompt's *content*, and
    that is captured as a hash in each adapter's provenance snapshot (see
    ``train.write_snapshot``). Keeping the path out of ``TrainConfig.fingerprint()``
    is what lets the fingerprint stay a claim about hyperparameters alone.
    """

    judge: str          # label-set name, e.g. "glm-5.2" — the ONE intended variable
    seed: int
    label_path: str
    out_dir: str
    config: TrainConfig = field(default_factory=TrainConfig)
    guard_prompt_path: str | None = None
    reproduction_mode: bool = False
