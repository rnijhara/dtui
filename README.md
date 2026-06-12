# dtui

A full-screen terminal app for diffusion language models.

Three ways in:

- **live** - type a prompt and watch a diffusion model denoise it in real time, in-process
- **viz** - replay a recorded denoising trajectory (no model, no GPU)
- **chat** - a lightweight coding agent over any OpenAI-compatible endpoint

Most LLMs write one token at a time, left to right. Diffusion models start from a
canvas of noise and denoise the whole thing in parallel, locking in the most
confident tokens each step. `live` runs the model in-process and streams every
denoising step to the canvas as it happens. `viz` replays a recording of that
same process anywhere, no GPU needed. `chat` is a normal lightweight coding
agent you can point at any endpoint.

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

### Modes

The app is modal, like a terminal editor. It opens in **command** mode, where
single keys run commands. Press `i` to enter **insert** mode (focuses the chat
input so you can type); `esc` returns to command mode. The footer always shows
the keys live in the current mode.

```
command mode   i insert · v chat/viz · space play/pause · r restart · q quit
insert  mode   esc commands
```

### live mode

Type a prompt and watch the model denoise it in real time. The model runs
in-process, so this needs the `[local]` extra and a GPU (run it on the GPU box
and SSH in):

```bash
dtui live                      # loads google/diffusiongemma-26B-A4B-it
dtui live --model <id>         # a different diffusion model
dtui live --mock               # fake provider, no GPU: preview the UI anywhere
```

Press `i`, type your prompt, watch it diffuse. `esc` back to command mode, `q`
to quit. No replay file, no timer: the model's own cadence drives the animation.

### viz mode

Replay a recorded denoising trajectory (the bundled sample, or your own JSONL):

```bash
dtui viz                       # plays the bundled sample trajectory
dtui viz path/to/run.jsonl     # plays your recording
dtui viz --fps 10              # faster playback
```

In command mode: `space` play/pause, `r` restart, `v` switch to chat, `q` quit.

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
