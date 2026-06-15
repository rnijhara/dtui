"""The full-screen dtui application.

Modal, like a terminal editor. The app starts in **command** mode: single keys
run commands (``v`` switch view, ``space`` play/pause, ``r`` restart, ``q``
quit). Press ``i`` to enter **insert** mode, which focuses the chat input so you
can type; ``esc`` returns to command mode. Two views live behind the modes: a
chat / coding-agent view and the denoising canvas, toggled with ``v``.

Insert mode implies the chat view (that is where the input lives); command mode
works in either view. The footer reflects the current mode, so the command keys
never collide with typing.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import ContentSwitcher, Footer, Header, Input

from dtui.agent.loop import AgentLoop
from dtui.providers.base import ChatProvider, TrajectoryProvider
from dtui.trajectory import Trajectory
from dtui.tui.canvas import CanvasView
from dtui.tui.chat import ChatView
from dtui.tui.viz import VizView


class DtuiApp(App):
    TITLE = "dtui"
    SUB_TITLE = "diffusion in your terminal"

    # Start in command mode: nothing is focused until the user presses ``i``.
    AUTO_FOCUS = None

    CSS = """
    ContentSwitcher { height: 1fr; }
    """

    BINDINGS = [
        # esc fires even while the input is focused (priority, App-down).
        Binding("escape", "enter_command", "commands", priority=True),
        Binding("i", "enter_insert", "insert"),
        Binding("v", "toggle_mode", "chat/viz"),
        Binding("space", "toggle_play", "play/pause"),
        Binding("r", "restart_viz", "restart"),
        # Custom action name so we never shadow the system ctrl+q quit binding.
        Binding("q", "cmd_quit", "quit"),
    ]

    command_mode: reactive[bool] = reactive(True)

    def __init__(
        self,
        *,
        trajectory: Trajectory | None = None,
        trajectory_provider: TrajectoryProvider | None = None,
        chat_provider: ChatProvider | None = None,
        agent: AgentLoop | None = None,
        start_mode: str = "chat",
        fps: float = 6.0,
        live_frame_delay: float = 0.08,
    ) -> None:
        super().__init__()
        self._trajectory = trajectory or Trajectory()
        self._trajectory_provider = trajectory_provider
        self._chat_provider = chat_provider
        self._agent = agent
        self._live = trajectory_provider is not None
        self._start_mode = start_mode if start_mode in ("chat", "viz") else "chat"
        self._interval = 1.0 / fps if fps > 0 else 0.16
        self._live_frame_delay = live_frame_delay
        self._play_timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        with ContentSwitcher(initial=self._start_mode, id="body"):
            yield ChatView(self._chat_provider, self._agent, id="chat")
            yield VizView(
                self._trajectory_provider,
                frame_delay=self._live_frame_delay,
                id="viz",
            )
        yield Footer()

    def on_mount(self) -> None:
        if self._trajectory.steps:
            self.viz.load(self._trajectory)
        if self._live:
            self.viz.hint = "press i, type a prompt, watch it diffuse"
            self.viz.refresh()
        self.sub_title = self._mode_label()
        if self._start_mode == "viz" and self._trajectory.steps:
            self.start_play()

    # -- convenient accessors (also used by tests) -------------------------

    @property
    def body(self) -> ContentSwitcher:
        return self.query_one("#body", ContentSwitcher)

    @property
    def viz(self) -> CanvasView:
        return self.query_one("#viz", VizView).canvas

    @property
    def mode(self) -> str:
        """The current view ("chat" or "viz")."""
        try:
            return self.body.current or self._start_mode
        except Exception:
            return self._start_mode

    def show_viz(self) -> None:
        self.body.current = "viz"

    def show_chat(self) -> None:
        self.body.current = "chat"

    # -- modal state -------------------------------------------------------

    def _mode_label(self) -> str:
        return "COMMAND" if self.command_mode else "INSERT"

    def watch_command_mode(self, value: bool) -> None:
        self.sub_title = self._mode_label()
        self.refresh_bindings()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Drive the footer: show only the keys that are live in this mode.

        Returns True (shown + enabled), False (hidden). The actions themselves
        are already gated by focus -- in insert mode the focused Input swallows
        printable keys before any App binding sees them -- so this only keeps the
        footer honest about what each key does right now.
        """
        if action == "enter_command":
            return not self.command_mode
        if action in ("enter_insert", "cmd_quit"):
            return self.command_mode
        if action == "toggle_mode":
            # In live mode there is no useful chat view to toggle to.
            return self.command_mode and not self._live
        if action in ("toggle_play", "restart_viz"):
            # Playback/restart only apply to a replayed trajectory, not a live stream.
            return self.command_mode and self.mode == "viz" and not self._live
        return True

    # -- actions -----------------------------------------------------------

    def action_enter_command(self) -> None:
        self.command_mode = True
        self.set_focus(None)
        # esc also stops a live stream: cancel the worker and clear "diffusing".
        try:
            self.query_one("#viz", VizView).stop()
        except Exception:
            pass

    def action_enter_insert(self) -> None:
        # Insert mode focuses an input. The live viz has its own prompt; every
        # other case types into the chat input, so fall back to the chat view.
        if self.mode == "viz" and not self._viz_has_input():
            self.show_chat()
        self.command_mode = False
        self.call_after_refresh(self._focus_input)

    def _viz_has_input(self) -> bool:
        try:
            self.query_one("#viz-prompt", Input)
            return True
        except Exception:
            return False

    def _focus_input(self) -> None:
        sel = "#viz-prompt" if (self.mode == "viz" and self._viz_has_input()) else "#prompt"
        try:
            self.query_one(sel, Input).focus()
        except Exception:
            pass

    def action_cmd_quit(self) -> None:
        self.exit()

    def action_toggle_mode(self) -> None:
        self.body.current = "viz" if self.mode == "chat" else "chat"
        # Leaving viz should not keep a playback timer running underneath.
        if self.mode != "viz":
            self.stop_play()
        self.refresh_bindings()

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
