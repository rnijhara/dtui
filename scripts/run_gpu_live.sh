#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -x ".venv/bin/dtui" ]]; then
  echo ".venv/bin/dtui not found. Run ./scripts/bootstrap_gpu_machine.sh first."
  exit 1
fi

if [[ -z "${HF_HOME:-}" && -d /workspace ]]; then
  export HF_HOME=/workspace/hf
fi

MODEL="${DTUI_LIVE_MODEL:-google/diffusiongemma-26B-A4B-it}"
MAX_NEW_TOKENS="${DTUI_MAX_NEW_TOKENS:-256}"
FRAME_DELAY="${DTUI_FRAME_DELAY:-0.25}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
fi

exec .venv/bin/dtui live \
  --model "$MODEL" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --frame-delay "$FRAME_DELAY"
