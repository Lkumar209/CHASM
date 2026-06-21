#!/usr/bin/env bash
# Regenerate headline table and Figures 1-4 from cached trajectories.
# Usage: bash scripts/reproduce.sh
set -euo pipefail

echo "=== CHASM reproduce.sh ==="
echo "Regenerating headline results from cached trajectories..."

# Check that trajectory cache exists
if [ ! -d "data/trajectories" ]; then
  echo "ERROR: data/trajectories/ not found. Run scripts/run_all.sh first."
  exit 1
fi

echo "TODO: implement at M11"
# uv run python scripts/build_dataset.py --from-cache
# uv run python -m chasm.eval.report --output results/
echo "=== Done ==="
