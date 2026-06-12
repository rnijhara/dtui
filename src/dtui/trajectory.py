"""The model-agnostic denoising trajectory format.

Every diffusion backend, however it exposes its internals, is normalized into a
stream of :class:`StepRecord` snapshots. One record is one denoising step: the
full token canvas at that step, plus optional per-position status and
confidence. The viz mode renders this stream; the JSONL helpers let us record a
real run once (e.g. on a GPU box) and replay it anywhere with no model present.

A position's ``status`` is one of:

* ``"masked"``    - not yet resolved (still noise / a mask token)
* ``"revealed"``  - resolved on this step (highlight it)
* ``"confirmed"`` - resolved on an earlier step and still standing

Status and confidence are optional. When a backend does not provide status, the
renderer derives "revealed this step" by diffing consecutive canvases, so every
backend gets the highlight effect for free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

MASKED = "masked"
REVEALED = "revealed"
CONFIRMED = "confirmed"


@dataclass
class StepRecord:
    """One denoising step: the whole canvas, not a delta."""

    step: int
    canvas: list[str]
    status: list[str] | None = None
    confidence: list[float] | None = None

    def __post_init__(self) -> None:
        n = len(self.canvas)
        if self.status is not None and len(self.status) != n:
            raise ValueError(
                f"status length {len(self.status)} != canvas length {n}"
            )
        if self.confidence is not None and len(self.confidence) != n:
            raise ValueError(
                f"confidence length {len(self.confidence)} != canvas length {n}"
            )

    def to_dict(self) -> dict:
        d: dict = {"step": self.step, "canvas": self.canvas}
        if self.status is not None:
            d["status"] = self.status
        if self.confidence is not None:
            d["confidence"] = self.confidence
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StepRecord":
        return cls(
            step=int(d["step"]),
            canvas=list(d["canvas"]),
            status=list(d["status"]) if d.get("status") is not None else None,
            confidence=(
                [float(c) for c in d["confidence"]]
                if d.get("confidence") is not None
                else None
            ),
        )

    def text(self) -> str:
        """Best-effort decoded text of the current canvas."""
        return "".join(self.canvas)


@dataclass
class Trajectory:
    """A finished or in-progress sequence of steps for one prompt."""

    prompt: str = ""
    steps: list[StepRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[StepRecord]:
        return iter(self.steps)

    def __getitem__(self, i: int) -> StepRecord:
        return self.steps[i]

    @property
    def final_text(self) -> str:
        return self.steps[-1].text() if self.steps else ""


def write_jsonl(path: str | Path, steps: Iterable[StepRecord], *, prompt: str = "") -> None:
    """Write a trajectory to JSONL. The first line is a header record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "header", "prompt": prompt}) + "\n")
        for s in steps:
            f.write(json.dumps(s.to_dict()) + "\n")


def read_jsonl(path: str | Path) -> Trajectory:
    """Read a trajectory written by :func:`write_jsonl`."""
    path = Path(path)
    prompt = ""
    steps: list[StepRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("type") == "header":
                prompt = d.get("prompt", "")
                continue
            steps.append(StepRecord.from_dict(d))
    return Trajectory(prompt=prompt, steps=steps)
