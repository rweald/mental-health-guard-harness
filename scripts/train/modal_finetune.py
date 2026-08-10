# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Run the LoRA fine-tune sweep on Modal.

Our nine runs (3 judges × 3 seeds) are independent and single-GPU-sized, so this fans
them out — each cell trains on its own GPU in parallel — reusing the
training code (``run_finetune``) unchanged and the frozen ``TrainConfig`` as-is
(paper-faithful shape; single-GPU per cell, no distributed training). Evaluation stays
local on the already-validated harness; only the small adapters come back, so the
licensed testset never leaves the local sandbox.

Design note: top-level imports here are ONLY ``modal`` — every use of our package lives
inside the remote functions, which run in the image where ``src`` is added. That is what
lets the ``modal`` CLI load this file locally without our package in its environment.

How to actually run it — prerequisites, the three commands, the first-run smoke test, and
pulling the adapters back for local eval — is in docs/running-the-finetune.md.
"""

from __future__ import annotations

import modal

# ── what varies (the experiment) ────────────────────────────────────────────
JUDGES = ["glm-4.6", "glm-5.2", "deepseek-v4-flash"]
SEEDS = (0, 1, 2)  # mirrors training.config.SEEDS (duplicated: config isn't importable
#                    in the local modal-launch env; the remote functions use the real one)

# Leaf directory a cell writes into the adapters volume. Mirrors the layout of
# training.train.adapter_dir, duplicated for the same reason as SEEDS; a test guards
# the two from drifting. Used locally to spot cells that already finished.
ADAPTER_NAME_FMT = "{judge}__seed{seed}"

# ── hardware + remote paths ─────────────────────────────────────────────────
GPU = "H100"
CACHE_DIR = "/cache"        # HF_HOME -> base model cached here (persistent volume)
DATA_DIR = "/data"          # dialogues + labels, baked into the image
ADAPTERS_DIR = "/adapters"  # trained adapters land here (persistent volume)

REQUIREMENTS = "scripts/train/modal-requirements.txt"

# Local path to the guard prompt baked into the image. Not in the repo; see
# paper_prompts and prompts/README.md for what belongs here.
GUARD_PROMPT_LOCAL = "prompts/guard.txt"

# ── image: CUDA torch + the lock-exported model-side deps + our source and data ────────
# torch is installed first, explicitly: the lock pins the CPU wheel (for local testing), so
# on a GPU we take torch from PyPI's default (CUDA) index instead. Everything else installs
# from a requirements file exported from uv.lock, so ALL transitive versions are pinned and
# can't drift. Regenerate that file whenever deps change (a test guards it staying current):
#   uv export --no-emit-project --no-dev --no-emit-package torch --no-hashes --frozen \
#       -o scripts/train/modal-requirements.txt
# torch is left unpinned until the first Modal run verifies a working CUDA build; pin it then.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch")
    .pip_install_from_requirements(REQUIREMENTS)
    .env({"PYTHONPATH": "/root/src", "HF_HOME": CACHE_DIR})
    .add_local_dir("src", "/root/src")
    .add_local_file("results/generated/dialogues.jsonl", f"{DATA_DIR}/dialogues.jsonl")
    .add_local_dir("results/generated/labels", f"{DATA_DIR}/labels")
    # The guard prompt is not in the repo (see paper_prompts) — it ships with the image
    # from the local file. Every cell of the sweep therefore renders its examples with
    # the identical prompt. Missing locally = the image build fails, which is the point.
    .add_local_file(GUARD_PROMPT_LOCAL, f"{DATA_DIR}/guard_prompt.txt")
)

app = modal.App("mindguard-inspired-si-detection-finetune", image=image)

# Persistent stores: base weights (downloaded once) and the trained adapters.
hf_cache = modal.Volume.from_name("mindguard-inspired-si-detection-hf-cache", create_if_missing=True)
adapters = modal.Volume.from_name("mindguard-inspired-si-detection-adapters", create_if_missing=True)


@app.function(volumes={CACHE_DIR: hf_cache}, timeout=3600)
def download_base():
    """One-time: cache the fine-tune base into the shared volume so runs start offline."""
    from huggingface_hub import snapshot_download

    from mh_guard_harness.training.config import TrainConfig

    model = TrainConfig().base_model
    path = snapshot_download(model)
    hf_cache.commit()
    print(f"cached {model} at {path}")


@app.function(
    gpu=GPU,
    timeout=2 * 3600,
    volumes={CACHE_DIR: hf_cache, ADAPTERS_DIR: adapters},
)
def finetune(judge: str, seed: int) -> dict:
    """Train one LoRA adapter (one judge × one seed) on a dedicated GPU, then persist it."""
    import os

    # The base weights are already in the cache volume (download_base). Pin the Hub to
    # offline so a transient network blip can't fail nine simultaneously-launched paid-GPU
    # cells on a live etag check — they load straight from the volume instead.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from mh_guard_harness.training.config import RunSpec
    from mh_guard_harness.training.train import adapter_dir, run_finetune

    out_dir = adapter_dir(judge, seed, root=ADAPTERS_DIR)
    run = RunSpec(
        judge=judge,
        seed=seed,
        label_path=f"{DATA_DIR}/labels/{judge}__full.jsonl",
        out_dir=out_dir,
        guard_prompt_path=f"{DATA_DIR}/guard_prompt.txt",
    )
    run_finetune(run, dialogues_path=f"{DATA_DIR}/dialogues.jsonl")
    adapters.commit()
    return {"judge": judge, "seed": seed, "out_dir": out_dir}


@app.local_entrypoint()
def main(skip_existing: bool = False):
    """Fan the 9 (judge, seed) cells out across GPUs; each trains in its own container.

    ``--skip-existing`` drops cells whose adapter is already in the volume. That resumes
    a sweep that partly failed, and keeps a cell run on its own (the GPU smoke) from
    being retrained — a rerun would overwrite the adapter with a numerically different
    one, invalidating any eval already computed against it.
    """
    cells = [(judge, seed) for judge in JUDGES for seed in SEEDS]
    if skip_existing:
        done = {entry.path.strip("/") for entry in adapters.listdir("/")}
        skipped = [c for c in cells if ADAPTER_NAME_FMT.format(judge=c[0], seed=c[1]) in done]
        cells = [c for c in cells if c not in skipped]
        # Name them: a silently shortened sweep looks identical to a complete one.
        print(f"skipping {len(skipped)} already in the volume: {sorted(skipped)}")
    print(f"launching {len(cells)} fine-tunes on {GPU}")
    for result in finetune.starmap(cells):
        print(f"done: {result}")
