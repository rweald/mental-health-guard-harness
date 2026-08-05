#!/usr/bin/env bash
# Score every fine-tuned adapter over the MindGuard testset, then report each one's
# metrics.
#
# Usage:
#   ./scripts/eval/score_all_adapters.sh           # skip cells already scored
#   FORCE=1 ./scripts/eval/score_all_adapters.sh   # rescore everything
#
# Two phases because they need different runtimes: scoring runs in the GPU container,
# metrics run in the local torch-free env (the container has no sklearn/scipy).
# Scoring is resumable — an existing non-empty scores file is left alone unless FORCE
# is set, so a run interrupted partway doesn't redo the cells that finished.
#
# Only `<judge>__seed<N>` adapters are scored. That deliberately excludes `_smoke`,
# which was trained locally on the GB10: adapters trained on different hardware differ
# numerically and must not enter the comparison (docs/running-the-finetune.md).
set -euo pipefail
cd "$(dirname "$0")/../.."

BASE=Qwen/Qwen3Guard-Gen-8B

# The guard prompt is supplied, not shipped (see prompts/README.md). It must be the
# same file the adapters were trained against, or the scores measure a prompt mismatch.
GUARD_PROMPT="${GUARD_PROMPT:-prompts/guard.txt}"
if [[ ! -f "$GUARD_PROMPT" ]]; then
  echo "guard prompt not found at $GUARD_PROMPT — set GUARD_PROMPT=<path> (see prompts/README.md)" >&2
  exit 1
fi

shopt -s nullglob
cells=(results/train/adapters/*__seed*/)
if ((${#cells[@]} == 0)); then
  echo "no adapters found under results/train/adapters/ — pull them from the Modal volume first" >&2
  exit 1
fi

failed=()

echo "== phase 1: scoring ${#cells[@]} adapters (GPU container) =="
for cell in "${cells[@]}"; do
  name=$(basename "$cell")
  out="results/eval/${name}_scores.jsonl"
  if [[ -s "$out" && -z "${FORCE:-}" ]]; then
    echo "-- $name: already scored, skipping ($out)"
    continue
  fi
  echo "-- $name"
  # Don't abort the whole sweep on one bad cell; collect and report at the end.
  if ! ./scripts/eval/run.sh python scripts/eval/score_testset.py \
      --model "$BASE" --adapter "${cell%/}" --output "$out" \
      --guard-prompt "$GUARD_PROMPT"; then
    echo "!! $name: scoring failed" >&2
    failed+=("$name")
  fi
done

echo
echo "== phase 2: metrics (local env) =="
for cell in "${cells[@]}"; do
  name=$(basename "$cell")
  out="results/eval/${name}_scores.jsonl"
  if [[ ! -s "$out" ]]; then
    echo "-- $name: no scores file, skipping"
    continue
  fi
  echo "=== $name ==="
  uv run python scripts/eval/report_metrics.py --scores "$out"
  echo
done

echo "== done in $((SECONDS / 60))m$((SECONDS % 60))s =="

if ((${#failed[@]} > 0)); then
  echo "FAILED to score: ${failed[*]}" >&2
  exit 1
fi
