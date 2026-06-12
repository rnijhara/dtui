"""The denoising canvas widget.

Renders one :class:`~dtui.trajectory.StepRecord` at a time as a grid of colored
token pieces. As you step through the trajectory, positions resolve out of
noise: masked cells are dim, a cell that resolved on this step lights up, and a
settled cell is plain. When a backend gives no per-position status, the widget
derives "resolved this step" by diffing against the previous canvas, so every
backend gets the highlight effect.
"""

from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual.widget import Widget

from dtui.trajectory import CONFIRMED, MASKED, REVEALED, StepRecord, Trajectory

_MASKED_STYLE = Style(color="grey37", dim=True)
_CONFIRMED_STYLE = Style(color="white")
_REVEALED_STYLE = Style(color="green3", bold=True)

# Confidence buckets, mirroring the LLaDA reference visualization.
_CONF_LOW = Style(color="red", bold=True)
_CONF_MID = Style(color="dark_orange", bold=True)
_CONF_HIGH = Style(color="green3", bold=True)


def _conf_style(c: float) -> Style:
    if c < 0.3:
        return _CONF_LOW
    if c < 0.7:
        return _CONF_MID
    return _CONF_HIGH


class CanvasView(Widget):
    """Displays a denoising trajectory, one step at a time."""

    DEFAULT_CSS = """
    CanvasView {
        padding: 1 2;
        height: 1fr;
        background: $surface;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.trajectory = Trajectory()
        self.index = 0

    # -- data / navigation -------------------------------------------------

    def load(self, trajectory: Trajectory) -> None:
        self.trajectory = trajectory
        self.index = 0
        self.refresh()

    @property
    def total(self) -> int:
        return len(self.trajectory)

    @property
    def at_end(self) -> bool:
        return self.index >= self.total - 1

    def current_record(self) -> StepRecord | None:
        if not self.trajectory.steps:
            return None
        return self.trajectory.steps[self.index]

    def advance(self) -> bool:
        """Move to the next step. Returns False if already at the end."""
        if self.at_end:
            return False
        self.index += 1
        self.refresh()
        return True

    def reset(self) -> None:
        self.index = 0
        self.refresh()

    # -- rendering ---------------------------------------------------------

    def _cell_style(self, rec: StepRecord, prev: StepRecord | None, i: int) -> Style:
        status = rec.status[i] if rec.status is not None else None
        if status is None:
            # Derive: changed since previous canvas => resolved this step.
            if prev is None or i >= len(prev.canvas):
                status = REVEALED
            elif rec.canvas[i] != prev.canvas[i]:
                status = REVEALED
            else:
                status = CONFIRMED
        if status == MASKED:
            return _MASKED_STYLE
        if status == REVEALED:
            if rec.confidence is not None and i < len(rec.confidence):
                return _conf_style(rec.confidence[i])
            return _REVEALED_STYLE
        return _CONFIRMED_STYLE

    def render(self) -> Text:
        rec = self.current_record()
        if rec is None:
            return Text("No trajectory loaded.", style="grey37")
        prev = self.trajectory.steps[self.index - 1] if self.index > 0 else None

        header = Text(
            f"denoising step {self.index + 1}/{self.total}\n\n", style="grey50"
        )
        body = Text()
        for i, piece in enumerate(rec.canvas):
            shown = piece if piece.strip() != "" else piece
            body.append(shown, style=self._cell_style(rec, prev, i))
        return header + body
