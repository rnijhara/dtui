# GPU Runbook

This runbook is for running `dtui live` on a GPU machine. It covers a generic
Linux GPU box and RunPod.

`dtui live` runs DiffusionGemma in-process. That means the machine needs enough
GPU memory for the model, a working NVIDIA driver, and the `[local]`
dependencies installed. The current default model is:

```text
google/diffusiongemma-26B-A4B-it
```

## Requirements

- Linux GPU machine with NVIDIA drivers visible through `nvidia-smi`
- Python 3.11+
- Git
- Enough disk for the model cache
- Enough VRAM for the model. H100 80GB is the tested setup.

If Hugging Face requires authentication for the model in your environment, run:

```bash
huggingface-cli login
```

## Fast Path

```bash
git clone https://github.com/rnijhara/dtui.git
cd dtui

./scripts/bootstrap_gpu_machine.sh
./scripts/run_gpu_live.sh
```

Inside the TUI:

- Press `i`
- Type a prompt
- Press `enter`
- Watch the active block denoise
- Press `esc` to return to command mode
- Press `q` to quit

## Manual Setup

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Install dependencies:

```bash
uv sync --extra local --extra dev
```

Verify CUDA from Python:

```bash
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

Run the focused tests that do not need the GPU model:

```bash
uv run pytest -q tests/test_live_viz.py tests/test_diffusion_gemma_adapter.py
```

Start live mode:

```bash
uv run dtui live --max-new-tokens 256 --frame-delay 0.25
```

Useful overrides:

```bash
uv run dtui live \
  --model google/diffusiongemma-26B-A4B-it \
  --max-new-tokens 512 \
  --frame-delay 0.25
```

`--max-new-tokens` controls the denoising canvas length. Higher values let the
model produce longer active blocks, but require more memory and can make frames
larger in the terminal.

`--frame-delay` slows down each visible frame. Use a larger value when recording
or inspecting the denoising process.

## RunPod

Start a RunPod instance with:

- H100 80GB if available
- A PyTorch/CUDA image
- Enough disk for the model cache
- SSH enabled

SSH in:

```bash
ssh root@<runpod-ip> -p <ssh-port> -i ~/.ssh/<key>
```

Optional `~/.ssh/config` entry:

```sshconfig
Host runpod
  HostName <runpod-ip>
  Port <ssh-port>
  User root
  IdentityFile ~/.ssh/<key>
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
```

Then:

```bash
ssh runpod
cd /workspace
git clone https://github.com/rnijhara/dtui.git
cd dtui

export HF_HOME=/workspace/hf
./scripts/bootstrap_gpu_machine.sh
./scripts/run_gpu_live.sh
```

For a persistent session:

```bash
tmux new -s dtui
export HF_HOME=/workspace/hf
./scripts/run_gpu_live.sh
```

Detach with `ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t dtui
```

## Local UI Preview Without GPU

Use the mock provider to verify the TUI layout anywhere:

```bash
uv sync --extra dev
uv run dtui live --mock
```

This does not run DiffusionGemma. It only exercises the live transcript and
canvas rendering.

## Troubleshooting

### `nvidia-smi` is missing

The machine does not have the NVIDIA runtime exposed. On RunPod, choose a CUDA
or PyTorch image and make sure the pod has a GPU attached.

### `torch.cuda.is_available()` is false

The Python environment cannot see CUDA. Confirm the NVIDIA driver works with
`nvidia-smi`, then reinstall the local dependencies:

```bash
uv sync --extra local --extra dev
```

### The first launch looks stuck

The model is probably downloading or loading. Keep `HF_HOME` on persistent disk
on GPU hosts:

```bash
export HF_HOME=/workspace/hf
```

### The denoising text moves in chunks

That is expected. DiffusionGemma uses block-style generation for long outputs:
one active block denoises, stable text is committed to the assistant transcript,
then the next block starts denoising.

### The TUI code changed but the running session did not

Restart `dtui live`. A running Python process does not pick up synced source
changes.
