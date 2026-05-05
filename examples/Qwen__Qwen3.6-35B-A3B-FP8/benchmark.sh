#!/usr/bin/env bash
set -euo pipefail

# Benchmark script for Qwen/Qwen3.6-35B-A3B-FP8
#
# Prerequisites:
#   pip install -e ".[dev]"
#   vllm must be installed and on PATH
#
# Usage:
#   # Full sweep (starts/stops server automatically):
#   bash examples/benchmark_qwen35b_fp8.sh
#
#   # Reuse an existing server (avoids cold-start per combo):
#   bash examples/benchmark_qwen35b_fp8.sh --no-server --port 8081
#
#   # Only plot existing results:
#   bash examples/benchmark_qwen35b_fp8.sh --plot-only
#
#   # Custom rates, concurrencies, port:
#   bash examples/benchmark_qwen35b_fp8.sh --rates "2.0 4.0 8.0" --concurrencies "1 4 8 16" --port 8081

MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
RESULTS_DIR="${RESULTS_DIR:-./results}"
INPUT_LEN="${INPUT_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
TIMEOUT="${TIMEOUT:-600}"
export PORT="${PORT:-8080}"

RATES="4.0 8.0 16.0 32.0"
CONCURRENCIES="1 2 4 8 16 32 64"
PLOT_ONLY=false
NO_SERVER=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rates) RATES="$2"; shift 2 ;;
        --concurrencies) CONCURRENCIES="$2"; shift 2 ;;
        --port) export PORT="$2"; shift 2 ;;
        --no-server) NO_SERVER=true; shift ;;
        --plot-only) PLOT_ONLY=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Derive experiment dir path from config values
MODEL_SAFE=$(echo "$MODEL" | tr '/' '__' | tr ':' '__')
EXPERIMENT="random_${INPUT_LEN}in_${OUTPUT_LEN}out"
SWEEP_DIR="${RESULTS_DIR}/${MODEL_SAFE}/${EXPERIMENT}"

echo "=== LM Benchmarks: $MODEL ==="
echo "Model:       $MODEL"
echo "Input len:   $INPUT_LEN"
echo "Output len:  $OUTPUT_LEN"
echo "Num prompts: $NUM_PROMPTS"
echo "Timeout:     ${TIMEOUT}s"
echo "Results:     $SWEEP_DIR"
echo ""

if $PLOT_ONLY; then
    echo "--- Plotting from existing results ---"
    benchmark plot "$SWEEP_DIR" --heatmap --tput-vs-tpu
    echo "Done. Plots saved to $SWEEP_DIR/plots/"
    exit 0
fi

# ---- Sweep ----
echo "--- Running sweep ---"
# Convert space-separated values to repeated flags (Click multiple=True)
RATE_FLAGS=()
for r in $RATES; do RATE_FLAGS+=(--rates "$r"); done
CONC_FLAGS=()
for c in $CONCURRENCIES; do CONC_FLAGS+=(--concurrencies "$c"); done

SWEEP_ARGS=(
    --model "$MODEL"
    "${RATE_FLAGS[@]}"
    "${CONC_FLAGS[@]}"
    --input-len "$INPUT_LEN"
    --output-len "$OUTPUT_LEN"
    --num-prompts "$NUM_PROMPTS"
    --timeout "$TIMEOUT"
)
if $NO_SERVER; then
    SWEEP_ARGS+=(--no-server)
fi
benchmark sweep "${SWEEP_ARGS[@]}"

echo ""
echo "=== Sweep complete ==="

# ---- Plot ----
echo ""
echo "--- Generating plots ---"
benchmark plot "$SWEEP_DIR" --heatmap --tput-vs-tpu

echo ""
echo "=== Done ==="
echo "Results: $SWEEP_DIR"
echo "Plots:   $SWEEP_DIR/plots/"
ls "$SWEEP_DIR"/*/run_metrics.json 2>/dev/null | wc -l | xargs echo "Completed runs:"
