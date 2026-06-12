"""Textual pilot test: drive viz mode through the bundled sample trajectory."""

from pathlib import Path

from dtui.trajectory import read_jsonl
from dtui.tui.app import DtuiApp

SAMPLE = Path(__file__).resolve().parent.parent / "src" / "dtui" / "data" / "sample_trajectory.jsonl"


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
