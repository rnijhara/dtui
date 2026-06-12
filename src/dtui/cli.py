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

    Two modes in one full-screen app:

      chat   a lightweight coding agent over any OpenAI-compatible endpoint

      viz    watch a diffusion model denoise a token canvas, step by step
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


def _bundled_sample() -> Path:
    return Path(str(resources.files("dtui.data") / "sample_trajectory.jsonl"))


if __name__ == "__main__":
    main()
