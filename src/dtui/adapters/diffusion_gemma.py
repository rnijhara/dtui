"""DiffusionGemma adapter (Google, open weights, HF Transformers).

DiffusionGemma denoises a fixed 256-token canvas. HF Transformers exposes the
per-step canvas through a push-based ``TextDiffusionStreamer``: the generation
loop calls ``put_draft(value)`` with the intermediate canvas each denoising step
and ``put(value)`` with confirmed tokens. We subclass it and override those
methods to capture each canvas as a :class:`~dtui.trajectory.StepRecord` instead
of printing.

The model itself (torch + transformers, the 26B weights) is heavy and lives
behind the ``[local]`` extra. It is imported lazily, so importing this module is
cheap and the capture logic can be unit-tested with a mocked streamer base.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Iterator

from dtui.trajectory import StepRecord

MODEL_ID = "google/diffusiongemma-26B-A4B-it"


# --- pure, dependency-free helpers (unit-testable without transformers) -----

def to_id_list(value: Any) -> list[int]:
    """Normalize a streamer ``value`` (tensor / ndarray / list) to a flat list.

    A 2D value (batch, seq) is reduced to its first row, matching how
    ``TextStreamer`` strips the batch dimension for a single sequence.
    """
    # torch.Tensor / numpy.ndarray both expose .tolist()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(x) for x in value]


def decode_canvas(token_ids: Any, tokenizer: Any) -> list[str]:
    """Decode a token-id canvas into one string piece per position."""
    ids = to_id_list(token_ids)
    pieces: list[str] = []
    for tid in ids:
        try:
            piece = tokenizer.decode([tid])
        except Exception:
            piece = "�"
        pieces.append(piece)
    return pieces


# --- the capturing streamer (subclasses transformers' base lazily) ----------

def make_capturing_streamer_class() -> type:
    """Build ``CapturingDiffusionStreamer`` against the installed transformers.

    Defined via a factory so the ``from transformers import ...`` happens only
    when a streamer is actually constructed. Tests substitute a stub
    ``transformers`` module to exercise the override logic without torch.
    """
    from transformers import TextDiffusionStreamer  # type: ignore

    class CapturingDiffusionStreamer(TextDiffusionStreamer):  # type: ignore[misc]
        """A TextDiffusionStreamer that records each canvas to a queue."""

        def __init__(self, tokenizer: Any, sink: "queue.Queue[StepRecord | None]", **kw: Any):
            super().__init__(tokenizer, **kw)
            self._sink = sink
            self._step = 0
            self._tok = tokenizer

        def put_draft(self, value: Any) -> None:  # intermediate canvas, per step
            canvas = decode_canvas(value, self._tok)
            self._sink.put(StepRecord(step=self._step, canvas=canvas))
            self._step += 1

        def put(self, value: Any) -> None:  # confirmed tokens (final / committed)
            canvas = decode_canvas(value, self._tok)
            self._sink.put(StepRecord(step=self._step, canvas=canvas))
            self._step += 1

        def end(self) -> None:
            self._sink.put(None)  # sentinel: trajectory complete

    return CapturingDiffusionStreamer


# --- the provider -----------------------------------------------------------

class DiffusionGemmaProvider:
    """In-process TrajectoryProvider for DiffusionGemma.

    Loads the model lazily and runs generation on a worker thread; the capturing
    streamer pushes one StepRecord per denoising step onto a queue that
    :meth:`stream_trajectory` drains.
    """

    name = "diffusiongemma"
    supports_trajectory = True

    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        max_new_tokens: int = 256,
        device_map: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import (  # type: ignore
                AutoProcessor,
                DiffusionGemmaForBlockDiffusion,
            )
        except ImportError as e:  # pragma: no cover - exercised only without [local]
            raise ImportError(
                "Running DiffusionGemma in-process needs the optional local "
                "dependencies. Install them with: pip install 'dtui[local]'"
            ) from e
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            self.model_id, dtype="auto", device_map=self.device_map
        )

    def stream_trajectory(self, prompt: str) -> Iterator[StepRecord]:
        self._ensure_loaded()
        assert self._model is not None and self._processor is not None
        streamer_cls = make_capturing_streamer_class()
        sink: "queue.Queue[StepRecord | None]" = queue.Queue()
        streamer = streamer_cls(self._processor.tokenizer, sink)

        messages = [{"role": "user", "content": prompt}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self._model.device)

        def _run() -> None:
            try:
                self._model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, streamer=streamer
                )
            finally:
                sink.put(None)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        while True:
            record = sink.get()
            if record is None:
                break
            yield record
        thread.join()
