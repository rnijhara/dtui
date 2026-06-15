"""The viz pane: a denoising canvas, optionally driven live by a provider.

Two ways to feed the canvas:

* **Replay** - the app pre-loads a recorded :class:`~dtui.trajectory.Trajectory`
  into the canvas and steps through it on a timer. No input box.
* **Live** - given a :class:`~dtui.providers.base.TrajectoryProvider`, this view
  shows a prompt input. On submit it runs the provider on a worker thread and
  pushes each :class:`~dtui.trajectory.StepRecord` to the canvas the instant it
  arrives, so you watch the model denoise your own prompt in real time.

The model's own cadence drives the animation: there is no replay file and no
timer. That is the whole point of ``dtui live``.

Cancellation matters here. A thread worker's OS thread cannot be force-killed,
so when a run is superseded (a new prompt) or stopped (esc / quit), we must (1)
signal the provider to stop producing via the worker's ``cancelled_event``, and
(2) drop any callbacks from a stale run via a generation counter, so steps from
an old prompt can never land on a new prompt's canvas.
"""

from __future__ import annotations

import time

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Input, Static
from textual.worker import Worker, get_current_worker

from dtui.providers.base import TrajectoryProvider
from dtui.trajectory import CONFIRMED, StepRecord
from dtui.tui.canvas import CanvasView


class VizView(Widget):
    DEFAULT_CSS = """
    VizView { height: 1fr; layout: vertical; }
    VizView #canvas { height: 1fr; }
    VizView #viz-transcript { height: 1fr; padding: 1 2; overflow-y: auto; }
    VizView #viz-prompt { dock: bottom; }
    VizView .live-message { width: 100%; height: auto; min-height: 1; margin-top: 1; }
    VizView .live-user { color: $accent; }
    VizView .live-canvas { width: 100%; height: auto; min-height: 1; margin-top: 1; }
    VizView .live-answer { width: 100%; height: auto; min-height: 1; margin-top: 0; }
    """

    def __init__(
        self,
        provider: TrajectoryProvider | None = None,
        *,
        frame_delay: float = 0.08,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.provider = provider
        self.frame_delay = max(0.0, frame_delay)
        self._gen = 0  # bumped per run; stale callbacks are dropped
        self._current_canvas: CanvasView | None = None
        self._current_answer: Static | None = None
        # Tests and app-level helpers ask for ``canvas`` before the first live
        # prompt. Keep an empty, unmounted fallback so those reads stay harmless.
        self._fallback_canvas = CanvasView(preview_only=True)

    def compose(self) -> ComposeResult:
        if self.provider is None:
            yield CanvasView(id="canvas")
        else:
            with VerticalScroll(id="viz-transcript"):
                yield Static(
                    "press i, type a prompt, watch it diffuse",
                    classes="live-message",
                    id="viz-hint",
                )
            yield Input(placeholder="Prompt  (enter to diffuse)", id="viz-prompt")

    @property
    def canvas(self) -> CanvasView:
        if self.provider is None:
            return self.query_one("#canvas", CanvasView)
        return self._current_canvas or self._fallback_canvas

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt or self.provider is None:
            return
        self.query_one("#viz-prompt", Input).value = ""
        self._gen += 1
        gen = self._gen
        if self._current_canvas is not None:
            self._current_canvas.finish_live()
        canvas, answer = await self._append_live_turn(prompt)
        canvas.begin_live(prompt)
        self._run(prompt, gen, canvas, answer)

    async def _append_live_turn(self, prompt: str) -> tuple[CanvasView, Static]:
        transcript = self.query_one("#viz-transcript", VerticalScroll)
        try:
            await self.query_one("#viz-hint", Static).remove()
        except Exception:
            pass
        await transcript.mount(
            Static(f"[b]you[/b]\n{escape(prompt)}", classes="live-message live-user")
        )
        canvas = CanvasView(preview_only=True, classes="live-canvas")
        await transcript.mount(canvas)
        answer = Static(self._answer_markup(""), classes="live-message live-answer")
        await transcript.mount(answer)
        self._current_canvas = canvas
        self._current_answer = answer
        self._scroll_to_end()
        return canvas, answer

    def stop(self) -> None:
        """Cancel any live stream and clear the diffusing state (e.g. on esc)."""
        self._gen += 1  # invalidate any in-flight callbacks
        self.workers.cancel_node(self)
        if self._current_canvas is not None:
            self._current_canvas.finish_live()

    def _run(self, prompt: str, gen: int, canvas: CanvasView, answer: Static) -> Worker:
        provider = self.provider

        def job() -> None:
            worker = get_current_worker()
            try:
                for record in provider.stream_trajectory(prompt, cancel=worker.cancelled_event):  # type: ignore[union-attr]
                    if worker.is_cancelled or self._gen != gen:
                        return
                    self._post(gen, self._push_live_record, canvas, answer, record)
                    if self.frame_delay:
                        time.sleep(self.frame_delay)
                self._post(gen, self._finish_live_turn, canvas, answer)
            except Exception as e:  # surface model/provider errors in the transcript
                self._post(gen, self._show_live_error, canvas, answer, e)

        # exclusive: starting a run cancels the previous one in this group.
        return self.run_worker(job, thread=True, exclusive=True, exit_on_error=False)

    def _push_live_record(
        self, canvas: CanvasView, answer: Static, record: StepRecord
    ) -> None:
        canvas.push_step(record)
        text = self._committed_text(record)
        if text is not None:
            answer.update(self._answer_markup(text))

    def _finish_live_turn(self, canvas: CanvasView, answer: Static) -> None:
        if not self._answer_has_text(answer):
            rec = canvas.current_record()
            if rec is not None and rec.text().strip():
                answer.update(self._answer_markup(rec.text().strip()))
        canvas.finish_live()
        canvas.display = False

    def _show_live_error(
        self, canvas: CanvasView, answer: Static, error: Exception
    ) -> None:
        message = f"{type(error).__name__}: {error}"
        canvas.show_error(message)
        answer.update(f"[red]{escape(message)}[/red]")

    def _committed_text(self, record: StepRecord) -> str | None:
        if len(record.canvas) == 1 and record.status == [CONFIRMED]:
            text = record.text().strip()
            return text if text else None
        return None

    def _answer_has_text(self, answer: Static) -> bool:
        content = str(answer.content)
        marker = "[b]assistant[/b]\n"
        return marker in content and bool(content.split(marker, 1)[1].strip())

    def _answer_markup(self, text: str) -> str:
        if not text:
            return "[grey50]────────────────[/grey50]\n[b]assistant[/b]"
        return f"[grey50]────────────────[/grey50]\n[b]assistant[/b]\n{escape(text)}"

    def _post(self, gen: int, fn, *args) -> None:
        """Apply a canvas update on the app thread.

        Drops the update if the run was superseded (``gen`` no longer current)
        and swallows the shutdown race where the event loop is already gone.
        """

        def apply() -> None:
            if gen == self._gen:
                fn(*args)
                self._scroll_to_end()

        try:
            self.app.call_from_thread(apply)
        except Exception:
            pass  # event loop gone (app exiting) -- nothing to update

    def _scroll_to_end(self) -> None:
        if self.provider is None:
            return
        try:
            transcript = self.query_one("#viz-transcript", VerticalScroll)
            self.call_after_refresh(
                transcript.scroll_end,
                animate=False,
                immediate=True,
                force=True,
                y_axis=True,
            )
        except Exception:
            pass
