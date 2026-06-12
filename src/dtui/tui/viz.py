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

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Input
from textual.worker import Worker, get_current_worker

from dtui.providers.base import TrajectoryProvider
from dtui.tui.canvas import CanvasView


class VizView(Widget):
    DEFAULT_CSS = """
    VizView { height: 1fr; layout: vertical; }
    VizView #canvas { height: 1fr; }
    VizView #viz-prompt { dock: bottom; }
    """

    def __init__(self, provider: TrajectoryProvider | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.provider = provider
        self._gen = 0  # bumped per run; stale callbacks are dropped

    def compose(self) -> ComposeResult:
        yield CanvasView(id="canvas")
        if self.provider is not None:
            yield Input(placeholder="Prompt  (enter to diffuse)", id="viz-prompt")

    @property
    def canvas(self) -> CanvasView:
        return self.query_one("#canvas", CanvasView)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt or self.provider is None:
            return
        self.query_one("#viz-prompt", Input).value = ""
        self._gen += 1
        self.canvas.begin_live(prompt)
        self._run(prompt, self._gen)

    def stop(self) -> None:
        """Cancel any live stream and clear the diffusing state (e.g. on esc)."""
        self._gen += 1  # invalidate any in-flight callbacks
        self.workers.cancel_node(self)
        self.canvas.finish_live()

    def _run(self, prompt: str, gen: int) -> Worker:
        canvas = self.canvas
        provider = self.provider

        def job() -> None:
            worker = get_current_worker()
            try:
                for record in provider.stream_trajectory(prompt, cancel=worker.cancelled_event):  # type: ignore[union-attr]
                    if worker.is_cancelled or self._gen != gen:
                        return
                    self._post(gen, canvas.push_step, record)
                self._post(gen, canvas.finish_live)
            except Exception as e:  # surface model/provider errors in the canvas
                self._post(gen, canvas.show_error, f"{type(e).__name__}: {e}")

        # exclusive: starting a run cancels the previous one in this group.
        return self.run_worker(job, thread=True, exclusive=True, exit_on_error=False)

    def _post(self, gen: int, fn, *args) -> None:
        """Apply a canvas update on the app thread.

        Drops the update if the run was superseded (``gen`` no longer current)
        and swallows the shutdown race where the event loop is already gone.
        """

        def apply() -> None:
            if gen == self._gen:
                fn(*args)

        try:
            self.app.call_from_thread(apply)
        except Exception:
            pass  # event loop gone (app exiting) -- nothing to update
