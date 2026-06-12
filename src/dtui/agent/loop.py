"""A minimal tool-calling agent loop.

Deliberately small: send the conversation plus tool schemas to the provider,
run any tool calls it returns, feed the results back, repeat until the model
answers in plain text. Yields :class:`AgentEvent` objects so the TUI can render
the back-and-forth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

from dtui.agent import tools

DEFAULT_SYSTEM = (
    "You are dtui, a concise terminal coding agent. Use the provided tools to "
    "inspect and edit files and run commands. Prefer small, verifiable steps. "
    "When the task is done, reply in plain text."
)


class CompletionProvider(Protocol):
    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict: ...


@dataclass
class AgentEvent:
    kind: str  # "tool_call" | "tool_result" | "final"
    text: str = ""
    name: str = ""
    data: dict = field(default_factory=dict)


class AgentLoop:
    def __init__(
        self,
        provider: CompletionProvider,
        *,
        system: str = DEFAULT_SYSTEM,
        max_steps: int = 12,
    ) -> None:
        self.provider = provider
        self.max_steps = max_steps
        self.messages: list[dict] = [{"role": "system", "content": system}]

    def run(self, user_message: str) -> Iterator[AgentEvent]:
        self.messages.append({"role": "user", "content": user_message})
        for _ in range(self.max_steps):
            msg = self.provider.complete(self.messages, tools.SCHEMAS)
            self.messages.append(_as_assistant(msg))
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                yield AgentEvent(kind="final", text=msg.get("content") or "")
                return
            for call in tool_calls:
                name, args, call_id = _parse_call(call)
                yield AgentEvent(kind="tool_call", name=name, data=args)
                result = tools.dispatch(name, args)
                self.messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": result}
                )
                yield AgentEvent(kind="tool_result", name=name, text=result)
        yield AgentEvent(
            kind="final", text="(stopped: reached max tool steps without a final answer)"
        )


def _as_assistant(msg: dict) -> dict:
    out: dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def _parse_call(call: dict) -> tuple[str, dict, str]:
    fn = call.get("function", {})
    name = fn.get("name", "")
    raw = fn.get("arguments", "{}")
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except json.JSONDecodeError:
        args = {}
    return name, args, call.get("id", "")
