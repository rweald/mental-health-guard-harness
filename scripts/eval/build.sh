#!/usr/bin/env bash
# Build (or rebuild) the GPU eval image.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec docker compose -f docker/eval/compose.yml build eval
