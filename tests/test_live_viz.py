"""Pilot tests for live viz: type a prompt, watch the canvas stream steps.

Driven by the mock provider (delay=0) so no GPU or model is involved; this
exercises the same wiring dtui live uses against DiffusionGemma.
"""

from dtui.providers.mock import MockDiffusionProvider
from dtui.tui.app import DtuiApp

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
