#!/usr/bin/env bash
set -euo pipefail

# Quick smoke test for Qwen/Qwen3.6-35B-A3B-FP8
# Assumes vllm server is already running (use --no-server).
#
# Usage:
#   bash examples/smoke_qwen35b_fp8.sh
#   bash examples/smoke_qwen35b_fp8.sh --port 8081
#   bash examples/smoke_qwen35b_fp8.sh --port 8081 --num-prompts 50

MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
INPUT_LEN="${INPUT_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
TIMEOUT="${TIMEOUT:-120}"

# Port is set via env var (no --port flag on `benchmark run`).
export PORT="${PORT:-8081}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) export PORT="$2"; shift 2 ;;
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
    --rate 4.0 \
    --concurrency 1 \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --timeout "$TIMEOUT" \
    --no-server

echo ""
echo "=== Smoke test complete ==="
