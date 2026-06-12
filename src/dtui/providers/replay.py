"""Replay a recorded trajectory as a TrajectoryProvider.

This is the backend that needs no GPU and no model. It reads a JSONL trajectory
(recorded from a real run, or synthesized) and replays the steps. It is what
lets us build and test the entire viz mode locally, and it doubles as the
capture-then-replay path for producing a clean recording.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from dtui.trajectory import StepRecord, Trajectory, read_jsonl


class ReplayProvider:
    """A TrajectoryProvider backed by a recorded :class:`Trajectory`."""

    supports_trajectory = True

    def __init__(self, trajectory: Trajectory, name: str = "replay") -> None:
        self.trajectory = trajectory
        self.name = name

    @classmethod
    def from_file(cls, path: str | Path, name: str = "replay") -> "ReplayProvider":
        return cls(read_jsonl(path), name=name)

    def stream_trajectory(self, prompt: str) -> Iterator[StepRecord]:
        # The prompt is ignored: a replay is a fixed recording.
        yield from self.trajectory.steps
