#!/usr/bin/env bash
set -euo pipefail

# Smoke test + sweep for Qwen/Qwen3-Omni-30B-A3B-Instruct
# Assumes vllm server is already running (use --no-server).
#
# Usage:
#   bash examples/Qwen__Qwen3-Omni/smoke.sh
#   bash examples/Qwen__Qwen3-Omni/smoke.sh --port 8082

MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
INPUT_LEN="${INPUT_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
TIMEOUT="${TIMEOUT:-120}"

export PORT="${PORT:-8082}"
RATES="8.0 16.0 32.0"
CONCURRENCIES="4 8 16 32 64 128"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) export PORT="$2"; shift 2 ;;
        --rates) RATES="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Smoke test: $MODEL ==="
echo "Port:        $PORT"
echo "Num prompts: $NUM_PROMPTS"
echo "Input len:   $INPUT_LEN"
echo "Output len:  $OUTPUT_LEN"
echo ""

benchmark run \
    --model "$MODEL" \
    --rate 8.0 \
    --concurrency 1 \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --timeout "$TIMEOUT" \
    --no-server

echo ""
echo "=== Smoke test passed. Starting sweep ==="

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
    --num-prompts 200
    --timeout 600
    --no-server
)
benchmark sweep "${SWEEP_ARGS[@]}"

echo ""
echo "=== Done ==="
