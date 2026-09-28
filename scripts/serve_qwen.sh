#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
exec vllm serve "$MODEL_PATH" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --served-model-name "${MODEL_ID:-qwen-vptd}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-model-len "${MAX_MODEL_LEN:-16384}" "$@"
