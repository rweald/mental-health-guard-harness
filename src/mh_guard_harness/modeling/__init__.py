# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Modeling: the only place torch and transformers may be imported.

All torch / transformers / trl / peft / accelerate usage is quarantined to this
subpackage so the rest of ``mh_guard_harness`` stays torch-free and
CPU-only. Concretely:

- Locally (on the Spark host), torch is the CPU wheel from the ``torch-cpu``
  dependency group. Modeling code is unit-tested here against tiny random dummy
  models — log-prob-at-label-position indexing, prompt/numbered-turn formatting,
  tokenizer behavior, scoring math — all in milliseconds, no GPU.
- The real 8B eval and the LoRA fine-tune run in the CUDA container (NGC base
  image, torch prebuilt for the GB10), not locally.

Anything importable outside this subpackage must not depend on torch.
"""
