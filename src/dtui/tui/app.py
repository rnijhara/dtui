"""The full-screen dtui application.

Two modes in one app, toggled with ``v``: a chat / coding-agent view (simple
mode) and the denoising canvas (viz mode). Viz playback runs on a timer; space
plays/pauses, ``r`` restarts.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import ContentSwitcher, Footer, Header

from dtui.agent.loop import AgentLoop
from dtui.providers.base import ChatProvider
from dtui.trajectory import Trajectory
from dtui.tui.canvas import CanvasView
from dtui.tui.chat import ChatView


class DtuiApp(App):
    TITLE = "dtui"
    SUB_TITLE = "diffusion in your terminal"

    CSS = """
    ContentSwitcher { height: 1fr; }
    """

    BINDINGS = [
        ("v", "toggle_mode", "chat/viz"),
        ("space", "toggle_play", "play/pause"),
        ("r", "restart_viz", "restart"),
        ("q", "quit", "quit"),
    ]

    def __init__(
        self,
        *,
        trajectory: Trajectory | None = None,
        chat_provider: ChatProvider | None = None,
        agent: AgentLoop | None = None,
        start_mode: str = "chat",
        fps: float = 6.0,
    ) -> None:
        super().__init__()
        self._trajectory = trajectory or Trajectory()
        self._chat_provider = chat_provider
        self._agent = agent
        self._start_mode = start_mode if start_mode in ("chat", "viz") else "chat"
        self._interval = 1.0 / fps if fps > 0 else 0.16
        self._play_timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        with ContentSwitcher(initial=self._start_mode, id="body"):
            yield ChatView(self._chat_provider, self._agent, id="chat")
            yield CanvasView(id="viz")
        yield Footer()

    def on_mount(self) -> None:
        if self._trajectory.steps:
            self.viz.load(self._trajectory)
        if self._start_mode == "viz" and self._trajectory.steps:
            self.start_play()

    # -- convenient accessors (also used by tests) -------------------------

    @property
    def body(self) -> ContentSwitcher:
        return self.query_one("#body", ContentSwitcher)

    @property
    def viz(self) -> CanvasView:
        return self.query_one("#viz", CanvasView)

    @property
    def mode(self) -> str:
        return self.body.current or self._start_mode

    def show_viz(self) -> None:
        self.body.current = "viz"

    def show_chat(self) -> None:
        self.body.current = "chat"

    # -- actions -----------------------------------------------------------

    def action_toggle_mode(self) -> None:
        self.body.current = "viz" if self.mode == "chat" else "chat"

    def action_restart_viz(self) -> None:
        self.viz.reset()

    def action_toggle_play(self) -> None:
        if self.mode != "viz":
            return
        if self._play_timer is not None:
            self.stop_play()
        else:
            self.start_play()

    # -- viz playback ------------------------------------------------------

    def start_play(self) -> None:
        if self._play_timer is None:
            self._play_timer = self.set_interval(self._interval, self._tick)

    def stop_play(self) -> None:
        if self._play_timer is not None:
            self._play_timer.stop()
            self._play_timer = None

    def _tick(self) -> None:
        if not self.viz.advance():
            self.stop_play()
