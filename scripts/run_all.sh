#!/usr/bin/env bash
# Run the full pipeline: generate trajectories, score monitors, run Breaker, produce report.
# Usage: bash scripts/run_all.sh [--smoke]
set -euo pipefail

SMOKE=""
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE="experiment=smoke"
fi

echo "=== CHASM run_all.sh ==="
echo "TODO: implement at M11"
# uv run chasm $SMOKE
echo "=== Done ==="
