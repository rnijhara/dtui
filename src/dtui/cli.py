"""The ``dtui`` command-line entry point."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import click

from dtui import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="dtui")
def main() -> None:
    """dtui - diffusion language models in your terminal.

    Three ways in:

      live   type a prompt and watch a diffusion model denoise it, in process

      viz    replay a recorded denoising trajectory (no model, no GPU)

      chat   a lightweight coding agent over any OpenAI-compatible endpoint
    """


@main.command()
@click.option("--base-url", default=None, help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
@click.option("--model", default=None, help="Model id to request.")
@click.option("--api-key", default=None, help="API key (or set DTUI_API_KEY).")
@click.option("--agent/--no-agent", default=False, help="Enable the tool-using coding agent.")
def chat(base_url: str | None, model: str | None, api_key: str | None, agent: bool) -> None:
    """Start simple mode: stream chat against a diffusion (or any) endpoint."""
    from dtui.agent.loop import AgentLoop
    from dtui.config import ChatConfig
    from dtui.providers.openai_chat import OpenAIChatProvider
    from dtui.tui.app import DtuiApp

    cfg = ChatConfig.resolve(base_url, model, api_key)
    provider = OpenAIChatProvider(cfg.base_url, cfg.model, cfg.api_key)
    loop = AgentLoop(provider) if agent else None
    DtuiApp(chat_provider=provider, agent=loop, start_mode="chat").run()


@main.command()
@click.argument("trajectory", required=False, type=click.Path(exists=True, dir_okay=False))
@click.option("--fps", default=6.0, show_default=True, help="Playback steps per second.")
def viz(trajectory: str | None, fps: float) -> None:
    """Start viz mode, replaying a recorded denoising TRAJECTORY (JSONL).

    With no path, plays the bundled sample trajectory.
    """
    from dtui.trajectory import read_jsonl
    from dtui.tui.app import DtuiApp

    path = Path(trajectory) if trajectory else _bundled_sample()
    traj = read_jsonl(path)
    DtuiApp(trajectory=traj, start_mode="viz", fps=fps).run()


@main.command()
@click.option("--model", default=None, help=f"Model id (default: {'google/diffusiongemma-26B-A4B-it'}).")
@click.option("--mock", is_flag=True, default=False, help="Fake provider, no GPU. Previews the live UI offline.")
@click.option("--max-new-tokens", default=256, show_default=True, help="Canvas length to denoise.")
def live(model: str | None, mock: bool, max_new_tokens: int) -> None:
    """Live viz: type a prompt and watch the in-process model denoise it.

    Loads a diffusion model in-process (needs the [local] extra and a GPU), so
    this is meant to run on the GPU box. Use --mock to preview the UI anywhere.
    """
    from dtui.tui.app import DtuiApp

    if mock:
        from dtui.providers.mock import MockDiffusionProvider

        provider = MockDiffusionProvider()
    else:
        from dtui.adapters.diffusion_gemma import MODEL_ID, DiffusionGemmaProvider

        provider = DiffusionGemmaProvider(model or MODEL_ID, max_new_tokens=max_new_tokens)

    DtuiApp(trajectory_provider=provider, start_mode="viz").run()


def _bundled_sample() -> Path:
    return Path(str(resources.files("dtui.data") / "sample_trajectory.jsonl"))


if __name__ == "__main__":
    main()
