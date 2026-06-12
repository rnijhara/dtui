"""Generate a deterministic sample denoising trajectory.

Simulates how a masked-diffusion model resolves a fixed canvas: every position
starts as noise, then tokens resolve out of order (not left to right) over a
number of steps, each newly resolved token carrying a confidence. The output is
a JSONL trajectory used by viz mode and the tests.

Run:  PYTHONPATH=src python3 scripts/make_sample_trajectory.py
"""

from __future__ import annotations

import random
from pathlib import Path

from dtui.trajectory import CONFIRMED, MASKED, REVEALED, StepRecord, write_jsonl

TARGET = (
    "Diffusion models do not write left to right . "
    "They start from pure noise and resolve every token at once ."
)
SEED = 7
STEPS = 12
OUT = Path("src/dtui/data/sample_trajectory.jsonl")


def _noise(rng: random.Random, width: int) -> str:
    chars = "abcdefghijklmnopqrstuvwxyz"
    return "".join(rng.choice(chars) for _ in range(max(2, width))) + " "


def build() -> list[StepRecord]:
    rng = random.Random(SEED)
    words = TARGET.split(" ")
    pieces = [w + " " for w in words]
    n = len(pieces)

    # A fixed, shuffled reveal order so tokens resolve out of left-to-right order.
    order = list(range(n))
    rng.shuffle(order)
    reveal_step = {pos: (1 + (rank * STEPS) // n) for rank, pos in enumerate(order)}

    records: list[StepRecord] = []
    for step in range(STEPS + 1):
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


def main() -> None:
    records = build()
    write_jsonl(OUT, records, prompt="Explain how diffusion language models generate text.")
    print(f"wrote {len(records)} steps -> {OUT}")
    print("final:", records[-1].text())


if __name__ == "__main__":
    main()
