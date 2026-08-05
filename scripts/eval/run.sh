#!/usr/bin/env bash
# Run a command inside the GPU eval container (NGC torch + model-side deps,
# shared weight cache mounted, repo bind-mounted at /work).
#
# Usage:
#   ./scripts/eval/run.sh                            # readiness check (default CMD)
#   ./scripts/eval/run.sh bash                       # interactive shell
#   ./scripts/eval/run.sh python path/to.py          # run a script
#   ./scripts/eval/run.sh --online python pull.py    # allow Hub fetches this run
#   ./scripts/eval/run.sh -e FOO=bar -- python ...   # raw `run` opts before `--`
#
# Weights/datasets normally load from the offline cache. `--online` flips both HF
# offline vars for a one-off pull. For any other `docker compose run` option (-e, -v,
# …) put it before a literal `--`; everything after `--` (or everything, with no `--`)
# is the command exec'd in the container.
set -euo pipefail
cd "$(dirname "$0")/../.."

run_opts=()

if [[ "${1:-}" == "--online" ]]; then
  run_opts+=(-e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0)
  shift
fi

# If a literal `--` is present, treat everything before it as `run` options.
saw_sep=
for arg in "$@"; do
  [[ "$arg" == "--" ]] && { saw_sep=1; break; }
done
if [[ -n "$saw_sep" ]]; then
  while [[ $# -gt 0 && "$1" != "--" ]]; do
    run_opts+=("$1")
    shift
  done
  shift   # drop the `--`
fi

exec docker compose -f docker/eval/compose.yml run --rm \
  ${run_opts[@]+"${run_opts[@]}"} eval "$@"
