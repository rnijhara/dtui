# dtui

A full-screen terminal app for diffusion language models.

Two modes in one app:

- **chat** - a lightweight coding agent over any OpenAI-compatible endpoint
- **viz** - watch a diffusion model denoise a token canvas, step by step, as words resolve out of noise instead of streaming left to right

Most LLMs write one token at a time, left to right. Diffusion models start from a
canvas of noise and denoise the whole thing in parallel, locking in the most
confident tokens each step. `viz` mode shows that process in the terminal. `chat`
mode is a normal, lightweight coding agent you can point at any endpoint,
including a diffusion model served over an OpenAI-compatible API.

## Install

```bash
uv tool install dtui          # light: chat + viz replay + remote viz
# or
pipx install dtui
```

The default install is light (no torch). To run an open diffusion model
**in-process** for live viz, add the local extra:

```bash
uv tool install "dtui[local]"   # pulls torch + transformers
```

## Usage

```bash
dtui --help
```

### viz mode

Replay a recorded denoising trajectory (the bundled sample, or your own JSONL):

```bash
dtui viz                       # plays the bundled sample trajectory
dtui viz path/to/run.jsonl     # plays your recording
dtui viz --fps 10              # faster playback
```

Keys: `space` play/pause, `r` restart, `v` switch to chat, `q` quit.

### chat mode

Point it at any OpenAI-compatible endpoint:

```bash
dtui chat --base-url http://localhost:8000/v1 --model my-model
dtui chat --agent              # enable the read/write/edit/bash tool loop
```

Config resolves from flags, then env (`DTUI_BASE_URL`, `DTUI_MODEL`,
`DTUI_API_KEY`), then `~/.config/dtui/config.toml`:

```toml
[chat]
base_url = "http://localhost:8000/v1"
model = "diffusiongemma"
api_key = "..."
```

## How it works

The UI knows nothing about any specific model. A backend is just a provider:

- `ChatProvider` streams assistant text (any OpenAI-compatible endpoint).
- `TrajectoryProvider` yields per-step canvas snapshots (`StepRecord`) for viz.

Adding a model means writing one small adapter. Built in:

| Model | How the trajectory is captured |
|-------|--------------------------------|
| DiffusionGemma | subclass of `transformers.TextDiffusionStreamer` (in-process) |
| Replay | reads a recorded JSONL trajectory (no model, no GPU) |

Viz mode needs per-step denoising states, which only exist when the model runs
in-process or the endpoint streams the trajectory (e.g. Mercury's `diffusing`
flag). For a normal remote endpoint, viz degrades to chat. This is why the heavy
local-inference dependencies live behind the `[local]` extra.

### Recording a trajectory

Run a diffusion model once (e.g. on a GPU box), capture its denoising steps to a
JSONL with `TrajectoryProvider` + `dtui.trajectory.write_jsonl`, then `dtui viz`
that file anywhere, no GPU required.

## Development

```bash
uv sync --extra dev
uv run pytest
PYTHONPATH=src python3 scripts/make_sample_trajectory.py   # regenerate the sample
```

## License

MIT
