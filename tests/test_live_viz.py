"""Pilot tests for live viz: type a prompt, watch the canvas stream steps.

Driven by the mock provider (delay=0) so no GPU or model is involved; this
exercises the same wiring dtui live uses against DiffusionGemma.
"""

import time

from dtui.providers.mock import MockDiffusionProvider
from dtui.trajectory import CONFIRMED, StepRecord
from dtui.tui.app import DtuiApp
from dtui.tui.canvas import CanvasView
from textual.containers import VerticalScroll
from textual.widgets import Static

MY_KEYS = {"escape", "i", "v", "space", "r", "q"}


def _mode_keys(app: DtuiApp) -> set[str]:
    return set(app.active_bindings) & MY_KEYS


async def _wait(app, pilot, *, min_steps: int) -> None:
    for _ in range(300):
        await pilot.pause()
        if app.viz.total >= min_steps and not app.viz.live:
            return
    raise AssertionError(f"stream stalled at {app.viz.total} steps (live={app.viz.live})")


async def _type_prompt(pilot, text: str) -> None:
    await pilot.press("i")          # command -> insert (focuses the viz prompt)
    await pilot.pause()
    await pilot.press(*list(text))
    await pilot.press("enter")


def _live_transcript(app: DtuiApp):
    return app.query_one("#viz-transcript", VerticalScroll).children


def _live_prompts(app: DtuiApp) -> list[str]:
    return [
        str(child.content)
        for child in _live_transcript(app)
        if isinstance(child, Static) and str(child.content).startswith("[b]you[/b]")
    ]


def _live_answers(app: DtuiApp) -> list[str]:
    return [
        str(child.content)
        for child in _live_transcript(app)
        if isinstance(child, Static) and "[b]assistant[/b]" in str(child.content)
    ]


async def test_live_streams_steps_to_canvas():
    provider = MockDiffusionProvider(steps=12, delay=0.0)
    app = DtuiApp(trajectory_provider=provider, start_mode="viz")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.viz.total == 0          # nothing until you ask

        await _type_prompt(pilot, "hello")
        await _wait(app, pilot, min_steps=13)

        assert app.viz.total == 13         # steps + 1 (the noise step)
        assert app.viz.live is False
        final = app.viz.current_record().text()
        assert "Diffusion models do not write left to right" in final

        children = list(_live_transcript(app))
        assert len(children) == 3
        assert isinstance(children[0], Static)
        assert str(children[0].content) == "[b]you[/b]\nhello"
        assert children[1] is app.viz
        assert "Diffusion models do not write left to right" in str(children[2].content)
        preview = app.viz.render().plain
        assert "denoising step" not in preview
        assert "Diffusion models" in preview
        assert app.viz.display is False


async def test_i_focuses_the_viz_prompt_in_live():
    app = DtuiApp(trajectory_provider=MockDiffusionProvider(delay=0.0), start_mode="viz")
    async with app.run_test() as pilot:
        await pilot.press("i")
        await pilot.pause()
        assert app.mode == "viz"           # stays in viz, unlike replay
        assert app.command_mode is False
        assert app.focused is not None and app.focused.id == "viz-prompt"


async def test_live_footer_hides_chat_and_playback():
    app = DtuiApp(trajectory_provider=MockDiffusionProvider(delay=0.0), start_mode="viz")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.command_mode is True
        # live mode: no chat toggle, no replay playback/restart.
        assert _mode_keys(app) == {"i", "q"}


async def test_second_prompt_supersedes_first():
    """A new prompt mid-stream cleanly replaces the old one; no interleaving."""
    app = DtuiApp(trajectory_provider=MockDiffusionProvider(steps=12, delay=0.03), start_mode="viz")
    async with app.run_test() as pilot:
        await _type_prompt(pilot, "first")
        for _ in range(40):
            await pilot.pause()
            if app.viz.total >= 2:
                break
        # Still in insert mode with the input focused: type a second prompt.
        await pilot.press(*list("second"))
        await pilot.press("enter")
        await _wait(app, pilot, min_steps=13)
        # Exactly one clean trajectory, not two interleaved (>13 would mean stale steps).
        assert app.viz.total == 13
        assert app.viz.live is False

        children = list(_live_transcript(app))
        canvases = [child for child in children if isinstance(child, CanvasView)]
        assert _live_prompts(app) == ["[b]you[/b]\nfirst", "[b]you[/b]\nsecond"]
        assert len(_live_answers(app)) == 2
        assert len(canvases) == 2
        assert canvases[0].live is False
        assert canvases[1] is app.viz


async def test_live_answer_ignores_non_monotonic_draft_text():
    class DraftThenCommitted:
        name = "draft-then-committed"
        supports_trajectory = True

        def stream_trajectory(self, prompt, cancel=None):
            yield StepRecord(step=0, canvas=["partial answer"])
            yield StepRecord(step=1, canvas=["completely different draft"])
            yield StepRecord(step=2, canvas=["final answer"], status=[CONFIRMED])

    app = DtuiApp(trajectory_provider=DraftThenCommitted(), start_mode="viz")
    async with app.run_test() as pilot:
        await _type_prompt(pilot, "x")
        await _wait(app, pilot, min_steps=3)

        answers = _live_answers(app)
        assert answers == [
            "[grey50]────────────────[/grey50]\n[b]assistant[/b]\nfinal answer"
        ]
        preview = app.viz.render().plain
        assert "denoising step" not in preview
        assert "final answer" in preview
        assert app.viz.display is False


async def test_live_preview_shows_draft_text_while_running():
    class SlowDraftThenCommitted:
        name = "slow-draft-then-committed"
        supports_trajectory = True

        def stream_trajectory(self, prompt, cancel=None):
            yield StepRecord(step=0, canvas=["partial answer"])
            time.sleep(0.2)
            yield StepRecord(step=1, canvas=["final answer"], status=[CONFIRMED])

    app = DtuiApp(trajectory_provider=SlowDraftThenCommitted(), start_mode="viz")
    async with app.run_test() as pilot:
        await _type_prompt(pilot, "x")
        for _ in range(200):
            await pilot.pause()
            if app.viz.total >= 1 and app.viz.live:
                break

        assert app.viz.live is True
        assert app.viz.display is True
        preview = app.viz.render().plain
        assert "partial answer" in preview
        assert "denoising step" not in preview
        assert _live_answers(app) == [
            "[grey50]────────────────[/grey50]\n[b]assistant[/b]"
        ]

        await _wait(app, pilot, min_steps=2)
        assert _live_answers(app) == [
            "[grey50]────────────────[/grey50]\n[b]assistant[/b]\nfinal answer"
        ]
        assert app.viz.display is False


async def test_live_preview_expands_for_long_draft_text():
    long_draft = " ".join(f"draftword{i}" for i in range(180)) + " tail-marker"

    class SlowLongDraft:
        name = "slow-long-draft"
        supports_trajectory = True

        def stream_trajectory(self, prompt, cancel=None):
            yield StepRecord(step=0, canvas=[long_draft])
            time.sleep(0.2)
            yield StepRecord(step=1, canvas=["final answer"], status=[CONFIRMED])

    app = DtuiApp(trajectory_provider=SlowLongDraft(), start_mode="viz")
    async with app.run_test(size=(100, 40)) as pilot:
        await _type_prompt(pilot, "x")
        for _ in range(200):
            await pilot.pause()
            if app.viz.total >= 1 and app.viz.live and app.viz.region.height > 6:
                break

        assert app.viz.live is True
        assert app.viz.display is True
        assert app.viz.region.height > 6
        preview = app.viz.render().plain
        assert "tail-marker" in preview
        assert "denoising step" not in preview


async def test_esc_stops_live_stream():
    app = DtuiApp(trajectory_provider=MockDiffusionProvider(steps=12, delay=0.03), start_mode="viz")
    async with app.run_test() as pilot:
        await _type_prompt(pilot, "hello")
        for _ in range(40):
            await pilot.pause()
            if app.viz.total >= 2:
                break
        await pilot.press("escape")
        await pilot.pause()
        assert app.command_mode is True
        assert app.viz.live is False
        frozen = app.viz.total
        for _ in range(15):
            await pilot.pause()
        assert app.viz.total == frozen      # stream stopped; no more steps land


async def test_live_surfaces_provider_errors():
    class Boom:
        name = "boom"
        supports_trajectory = True

        def stream_trajectory(self, prompt, cancel=None):
            raise RuntimeError("kaboom")
            yield  # unreachable; makes this a generator

    app = DtuiApp(trajectory_provider=Boom(), start_mode="viz")
    async with app.run_test() as pilot:
        await _type_prompt(pilot, "x")
        for _ in range(200):
            await pilot.pause()
            if app.viz.error:
                break
        assert app.viz.error and "kaboom" in app.viz.error
        assert app.viz.live is False
