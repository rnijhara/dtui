"""The two provider protocols the TUI renders against.

The UI knows nothing about any specific model. A backend is just an object that
implements one or both of these protocols:

* :class:`ChatProvider` - streams assistant text for simple mode. Any
  OpenAI-compatible endpoint satisfies this.
* :class:`TrajectoryProvider` - yields :class:`~dtui.trajectory.StepRecord`
  snapshots for viz mode. Only backends that expose per-step denoising state can
  do this: a model running in-process, a recorded replay, or a hosted API that
  streams the trajectory (e.g. Mercury's ``diffusing`` flag).

Adding a new diffusion model means writing one small adapter, not touching the
UI. ``supports_trajectory`` lets the UI degrade viz mode to chat mode when a
backend can only stream final text.
"""

from __future__ import annotations

from typing import Iterator, Protocol, Sequence, runtime_checkable

from dtui.trajectory import StepRecord

Message = dict[str, str]


@runtime_checkable
class ChatProvider(Protocol):
    """Streams assistant text deltas for a conversation."""

    name: str

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        """Yield text deltas for the assistant's reply to ``messages``."""
        ...


@runtime_checkable
class TrajectoryProvider(Protocol):
    """Yields per-step denoising snapshots for a single prompt."""

    name: str
    supports_trajectory: bool

    def stream_trajectory(self, prompt: str) -> Iterator[StepRecord]:
        """Yield one :class:`StepRecord` per denoising step."""
        ...
