#!/usr/bin/env bash
# Regenerate all CHASM results from scratch using EchoBackend (smoke) or a real backend.
#
# Usage:
#   bash scripts/reproduce.sh                      # smoke run (echo backend)
#   CHASM_BACKEND=api bash scripts/reproduce.sh    # full run with API backend
#
# Outputs:
#   results/metrics.json        per-condition monitor metrics
#   results/REPORT.md           hypothesis verdicts H1-H8
#   results/figures.json        figure data specs
#   data/hf_export/             HuggingFace dataset export

set -euo pipefail

BACKEND="${CHASM_BACKEND:-echo}"
N_SEEDS="${CHASM_N_SEEDS:-1}"
N_TASKS="${CHASM_N_TASKS:-2}"
RESULTS_DIR="${CHASM_RESULTS_DIR:-results}"
DATA_DIR="${CHASM_DATA_DIR:-data}"

echo "=== CHASM reproduce.sh ==="
echo "Backend:   ${BACKEND}"
echo "Seeds:     ${N_SEEDS}"
echo "Tasks/fam: ${N_TASKS}"
echo "Results:   ${RESULTS_DIR}"
echo ""

mkdir -p "${RESULTS_DIR}" "${DATA_DIR}/trajectories"

# ── Step 1: Build dataset ────────────────────────────────────────────────────
echo "[1/5] Building dataset..."
uv run python scripts/build_dataset.py \
    --tasks-per-family "${N_TASKS}" \
    --output "${DATA_DIR}/tasks.json"

# ── Step 2: Run eval ladder (all conditions, all monitors) ────────────────────
echo "[2/5] Running eval ladder..."
uv run python -m chasm.cli run-ladder \
    --backend "${BACKEND}" \
    --n-seeds "${N_SEEDS}" \
    --n-tasks "${N_TASKS}" \
    --output-dir "${RESULTS_DIR}"

# ── Step 3: Run Breaker adaptive red-team ────────────────────────────────────
echo "[3/5] Running Breaker adaptive red-team..."
uv run python -m chasm.cli run-breaker \
    --backend "${BACKEND}" \
    --n-rounds 2 \
    --output-dir "${RESULTS_DIR}/breaker"

# ── Step 4: Generate report and figure specs ─────────────────────────────────
echo "[4/5] Generating M11 report..."
uv run python -c "
from chasm.eval.report import run_report
run_report('${RESULTS_DIR}', '${RESULTS_DIR}')
"

# ── Step 5: Export HuggingFace dataset ───────────────────────────────────────
echo "[5/5] Exporting HuggingFace dataset..."
uv run python scripts/export_hf_dataset.py \
    --trajectories "${DATA_DIR}/trajectories" \
    --output "${DATA_DIR}/hf_export"

echo ""
echo "=== Done ==="
echo "Results written to ${RESULTS_DIR}/"
echo "  REPORT.md          — hypothesis verdicts"
echo "  figures.json       — figure data specs"
echo "  metrics.json       — per-condition metrics"
echo "HF dataset written to ${DATA_DIR}/hf_export/"
