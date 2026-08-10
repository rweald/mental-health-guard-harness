# Running the fine-tune sweep

The Step-4 LoRA fine-tune trains one Qwen3Guard-8B adapter per label set and seed —
**three judges × three seeds = nine runs**. The runs are independent and each fits on a
single GPU, so we run them on **Modal** (cloud GPUs), where they fan out and train in
parallel. This local box is too slow for the full sweep.

**Evaluation stays local**, on this box, on the already-validated harness. Only the small
adapters come back from Modal; the licensed testset never leaves the local sandbox. That
keeps the two lineages separate: our generated data + labels go out to train, the testset
stays home to score.

## Prerequisites

- The `modal` CLI, installed globally (isolated from any project venv):

  ```bash
  uv tool install modal
  ```

- A one-time Modal login (opens a browser to link your account):

  ```bash
  modal token new
  ```

Run all `modal` commands **from the repo root** — the app bakes `src/`, the generated
data, and the guard prompt into the image via paths relative to the root.

You also need a guard prompt at `prompts/guard.txt` (the harness ships none — see
[`prompts/README.md`](../prompts/README.md)). It is baked into the image so every cell
renders its training examples with the identical prompt; if it is missing, the image build
fails rather than training nine adapters against nothing. Point `GUARD_PROMPT_LOCAL` in
`scripts/train/modal_finetune.py` elsewhere if you keep it somewhere else.

## The three steps

```bash
# 1. One-time: cache the base model into a persistent Modal volume so runs start offline.
#    Also doubles as the first smoke test (see below).
modal run scripts/train/modal_finetune.py::download_base

# 2. The sweep: launch all nine cells at once; each trains on its own GPU in parallel.
modal run scripts/train/modal_finetune.py

# 3. Back on this box: pull the trained adapters into the training lineage.
modal volume get mindguard-inspired-si-detection-adapters / ./results/train/adapters/
```

After step 3 you'll have `results/train/adapters/<judge>__seed<N>/` for all nine cells —
each an adapter (`adapter_model.safetensors` + `adapter_config.json`), a provenance
snapshot (`run.json`, including the config fingerprint and the guard prompt's SHA-256),
and the tokenizer.

Score each adapter with the same prompt it was trained on; `run.json`'s `guard_prompt.sha256`
is what proves which one that was.

## First run is the smoke test

The image pins `transformers` / `trl` / `peft` to the versions the local runs used, but
the standard CUDA `torch` wheel running on Modal's H100 can only be verified by actually
running it. So treat the **first `download_base` plus a single cell** as the smoke test
before launching the whole sweep. To run just one cell, temporarily point the entrypoint at
a single `(judge, seed)` — or call the function directly:

```bash
modal run scripts/train/modal_finetune.py::finetune --judge glm-5.2 --seed 0
```

Watch for a clean loss curve and a saved adapter, exactly as in the local smoke.

## Then: evaluate locally

Scoring the pulled adapters happens on this box with the existing eval harness — load the
base model plus the adapter (`peft`), then run the testset scoring. (That eval path is
added separately; this doc covers getting the trained adapters here.)

## Notes

- **Same hardware for all nine.** Adapters trained on Modal's GPU differ numerically from
  ones trained here; that's fine as long as the whole sweep runs on Modal. Don't mix a
  locally-trained adapter into the comparison.
- **Cost.** Nine H100-hours-ish total, billed per second, nothing while idle. Verify current
  Modal pricing before launching.
- **Config is frozen.** The sweep uses `TrainConfig` as-is (3 epochs, global batch 128,
  cosine + warmup, assistant-only loss, 4096 context; LoRA learning-rate band). The judge is
  the only variable across cells. `TrainConfig.fingerprint()` is recorded per run so the
  cells can be proven to have shared hyperparameters; the guard prompt hash covers the
  prompt the same way.
