# Running the eval

Evaluation scores a model over the clinician-labeled MindGuard testset and reports the
validation metrics. It runs on this box in the GPU container — the evaluation lineage stays
local, and the testset is **scoring-only** (never trained on).

Two stages:

1. **Score** (container / GPU) — `score_testset.py` runs teacher-forced single-token scoring
   over every testset row and writes per-turn scores (`{index, p_unsafe, label,
   annotator_labels}`) into `results/eval/`.
2. **Metrics** (local / torch-free) — `report_metrics.py` reads that scores file and computes
   AUROC / FPR@TPR etc. (the container has torch but not sklearn/scipy).

## Prerequisites

- The eval image built: `./scripts/eval/build.sh`
- The model + testset present in the shared HF cache (pull once with `--online` if needed).
- A guard prompt file. The harness ships none — see [`prompts/README.md`](../prompts/README.md).
  Every command below passes its path explicitly; there is no default.

## Score a published model (baseline)

```bash
# smoke first (16 rows)
./scripts/eval/run.sh python scripts/eval/score_testset.py \
    --guard-prompt prompts/guard.txt --limit 16

# full testset
./scripts/eval/run.sh python scripts/eval/score_testset.py \
    --guard-prompt prompts/guard.txt
```

Add `--reproduction-mode` to require the prompt to hash-match the published reference
prompt; without it any prompt carrying the `{conversation}` placeholder is accepted, and
its hash is recorded in the run's provenance sidecar.

## Score a fine-tuned LoRA adapter

Point `--model` at the base and `--adapter` at an adapter pulled back from the sweep, and
give each run its own output file:

```bash
./scripts/eval/run.sh python scripts/eval/score_testset.py \
    --model Qwen/Qwen3Guard-Gen-8B \
    --adapter results/train/adapters/glm-5.2__seed0 \
    --output results/eval/glm-5.2__seed0_scores.jsonl \
    --guard-prompt prompts/guard.txt
```

**Score adapters with the prompt they were trained on.** A model trained against one guard
prompt and scored against another measures the mismatch. Each adapter's `run.json` records
the SHA-256 of the prompt that rendered its training examples; the scores file gets a
matching `.provenance.json`. Compare the two hashes if there is any doubt.

The fine-tune base shares the published model's decision tokens, so scoring needs no other
change. Repeat per adapter (three judges × three seeds).

## Compute the metrics (local)

```bash
uv run python scripts/eval/report_metrics.py \
    --scores results/eval/glm-5.2__seed0_scores.jsonl
```

## Notes

- **Scores land in `results/eval/`** — the evaluation lineage, never the training lineage.
- **The same harness scores every model**, so the published baseline and all fine-tuned
  adapters are directly comparable.
