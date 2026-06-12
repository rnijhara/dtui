"""Textual pilot tests: viz playback and the insert/command modal model."""

from pathlib import Path

from dtui.trajectory import read_jsonl
from dtui.tui.app import DtuiApp

SAMPLE = Path(__file__).resolve().parent.parent / "src" / "dtui" / "data" / "sample_trajectory.jsonl"

# The modal keys the app owns; system bindings (ctrl+q, ctrl+c, ...) are ignored.
MY_KEYS = {"escape", "i", "v", "space", "r", "q"}


def _mode_keys(app: DtuiApp) -> set[str]:
    """The app's own keys currently visible in the footer (check_action True/None)."""
    return set(app.active_bindings) & MY_KEYS


async def test_viz_plays_sample_trajectory():
    traj = read_jsonl(SAMPLE)
    assert len(traj) > 1

    app = DtuiApp(trajectory=traj, start_mode="chat")
    async with app.run_test() as pilot:
        # Toggle into viz mode.
        app.show_viz()
        await pilot.pause()
        assert app.mode == "viz"
        assert app.viz.total == len(traj)

        # Step 0 is noise: it should not yet contain the final sentence.
        assert "left to right" not in app.viz.render().plain

        # Advance through every denoising step.
        while app.viz.advance():
            pass

        rec = app.viz.current_record()
        assert rec is not None
        assert rec.text() == traj.final_text
        # The fully-denoised canvas renders the real sentence.
        assert "Diffusion models do not write left to right" in app.viz.render().plain


async def test_toggle_mode_action():
    traj = read_jsonl(SAMPLE)
    app = DtuiApp(trajectory=traj, start_mode="chat")
    async with app.run_test() as pilot:
        assert app.mode == "chat"
        await pilot.press("v")
        assert app.mode == "viz"
        await pilot.press("v")
        assert app.mode == "chat"


async def test_starts_in_command_mode():
    """Both views start in command mode with nothing focused (AUTO_FOCUS off)."""
    traj = read_jsonl(SAMPLE)
    for start in ("chat", "viz"):
        app = DtuiApp(trajectory=traj, start_mode=start)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.command_mode is True
            assert app.focused is None
            assert app.sub_title == "COMMAND"


async def test_i_enters_insert_and_focuses_prompt():
    traj = read_jsonl(SAMPLE)
    app = DtuiApp(trajectory=traj, start_mode="viz")
    async with app.run_test() as pilot:
        await pilot.press("i")
        await pilot.pause()
        assert app.command_mode is False
        assert app.mode == "chat"          # insert implies the chat view
        assert app.focused is not None
        assert app.focused.id == "prompt"
        assert app.sub_title == "INSERT"


async def test_escape_returns_to_command():
    traj = read_jsonl(SAMPLE)
    app = DtuiApp(trajectory=traj, start_mode="chat")
    async with app.run_test() as pilot:
        await pilot.press("i")
        await pilot.pause()
        assert app.command_mode is False
        await pilot.press("escape")
        await pilot.pause()
        assert app.command_mode is True
        assert app.focused is None


async def test_command_keys_type_in_insert_mode():
    """The crux: in insert mode the focused input swallows command keys."""
    from textual.widgets import Input

    traj = read_jsonl(SAMPLE)
    app = DtuiApp(trajectory=traj, start_mode="chat")
    async with app.run_test() as pilot:
        await pilot.press("i")
        await pilot.pause()
        await pilot.press("v", "r", "q")   # would be commands in command mode
        await pilot.pause()
        assert app.mode == "chat"          # no view toggle happened
        assert app.command_mode is False   # still typing
        assert app.query_one("#prompt", Input).value == "vrq"


async def test_footer_reflects_mode_and_view():
    traj = read_jsonl(SAMPLE)
    app = DtuiApp(trajectory=traj, start_mode="chat")
    async with app.run_test() as pilot:
        await pilot.pause()
        # command mode, chat view: viz-only keys hidden, esc hidden.
        assert _mode_keys(app) == {"i", "v", "q"}

        await pilot.press("v")             # -> viz view, still command mode
        await pilot.pause()
        assert _mode_keys(app) == {"i", "v", "space", "r", "q"}

        await pilot.press("i")             # -> insert mode (back to chat)
        await pilot.pause()
        assert _mode_keys(app) == {"escape"}
