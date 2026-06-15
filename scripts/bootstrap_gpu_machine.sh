#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ "${DTUI_WITH_DEV:-1}" == "1" ]]; then
  uv sync --extra local --extra dev
else
  uv sync --extra local
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "warning: nvidia-smi not found on PATH"
fi

uv run python - <<'PY'
import sys

import torch

print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device: {torch.cuda.get_device_name(0)}")
else:
    sys.exit("CUDA is not available to torch. Check the GPU runtime and NVIDIA drivers.")
PY

echo
echo "GPU setup complete."
echo "Run: ./scripts/run_gpu_live.sh"
