"""A fake TrajectoryProvider for developing and previewing the live viz.

This runs no model and needs no GPU. It synthesizes a plausible masked-diffusion
trajectory (positions start as noise, then resolve out of left-to-right order)
and yields the steps with a small delay so the live UI animates exactly as it
would against a real backend.

It is a development tool, not the post asset. ``dtui live`` uses the real
``DiffusionGemmaProvider``; ``dtui live --mock`` uses this so the streaming UI
can be exercised offline.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Iterator

from dtui.trajectory import CONFIRMED, MASKED, REVEALED, StepRecord

# A fixed illustrative answer. The prompt only seeds the reveal order, so the
# same sentence resolves in a different pattern each time, which is plenty for
# exercising the UI.
_RESPONSE = (
    "Diffusion models do not write left to right . "
    "They denoise the whole canvas at once , locking in confident tokens first ."
)

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _noise(rng: random.Random, width: int) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(max(2, width))) + " "


def build_trajectory(prompt: str, steps: int = 12, seed: int = 7) -> list[StepRecord]:
    """Synthesize a denoising trajectory toward ``_RESPONSE``."""
    rng = random.Random(f"{seed}:{prompt}")
    words = _RESPONSE.split(" ")
    pieces = [w + " " for w in words]
    n = len(pieces)

    order = list(range(n))
    rng.shuffle(order)
    reveal_step = {pos: (1 + (rank * steps) // n) for rank, pos in enumerate(order)}

    records: list[StepRecord] = []
    for step in range(steps + 1):
        canvas: list[str] = []
        status: list[str] = []
        confidence: list[float] = []
        for pos in range(n):
            rs = reveal_step[pos]
            if step < rs:
                canvas.append(_noise(rng, len(words[pos])))
                status.append(MASKED)
                confidence.append(0.0)
            elif step == rs:
                canvas.append(pieces[pos])
                status.append(REVEALED)
                confidence.append(round(rng.uniform(0.45, 0.99), 2))
            else:
                canvas.append(pieces[pos])
                status.append(CONFIRMED)
                confidence.append(1.0)
        records.append(
            StepRecord(step=step, canvas=canvas, status=status, confidence=confidence)
        )
    return records


class MockDiffusionProvider:
    """A TrajectoryProvider that fakes denoising, for offline UI testing."""

    name = "mock-diffusion"
    supports_trajectory = True

    def __init__(self, steps: int = 12, delay: float = 0.12, seed: int = 7) -> None:
        self.steps = steps
        self.delay = delay
        self.seed = seed

    def stream_trajectory(
        self, prompt: str, cancel: threading.Event | None = None
    ) -> Iterator[StepRecord]:
        for record in build_trajectory(prompt, self.steps, self.seed):
            if cancel is not None and cancel.is_set():
                return
            if self.delay:
                time.sleep(self.delay)
            yield record
