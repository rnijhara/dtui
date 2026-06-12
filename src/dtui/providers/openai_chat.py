"""An OpenAI-compatible chat provider over httpx.

This is the universal "simple mode" backend: it talks to any endpoint that
speaks the OpenAI ``/v1/chat/completions`` shape (vLLM, SGLang, LM Studio,
LocalAI, Mercury, OpenRouter, ...). It exposes:

* :meth:`stream` - yield text deltas (plain chat).
* :meth:`complete` - one tool-aware, non-streaming completion, used by the
  agent loop to decide tool calls.

Only the standard library plus httpx is used, so this works on the light
install with no torch.
"""

from __future__ import annotations

import json
from typing import Any, Iterator, Sequence

import httpx

Message = dict[str, Any]


class OpenAIChatProvider:
    name = "openai"
    supports_trajectory = False

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        timeout: float = 120.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        # extra_body carries provider-specific fields (e.g. Mercury's "diffusing")
        # that standard SDKs would strip.
        self.extra_body = extra_body or {}

    # -- internals ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _body(self, messages: Sequence[Message], **overrides: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": list(messages)}
        body.update(self.extra_body)
        body.update(overrides)
        return body

    # -- streaming chat ----------------------------------------------------

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        """Yield assistant text deltas using SSE streaming."""
        body = self._body(messages, stream=True)
        with httpx.stream(
            "POST", self._url(), headers=self._headers(), json=body, timeout=self.timeout
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                delta = _parse_sse_delta(line)
                if delta is _DONE:
                    return
                if delta:
                    yield delta

    # -- tool-aware single completion -------------------------------------

    def complete(
        self, messages: Sequence[Message], tools: list[dict] | None = None
    ) -> dict[str, Any]:
        """Return one non-streaming completion message.

        The returned dict has the OpenAI assistant-message shape: ``content``
        (str or None) and optional ``tool_calls`` (list).
        """
        body = self._body(messages)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        resp = httpx.post(
            self._url(), headers=self._headers(), json=body, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]


_DONE = object()


def _parse_sse_delta(line: str) -> Any:
    """Parse one SSE line into a text delta, ``_DONE``, or ``None``."""
    if not line:
        return None
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if payload == "[DONE]":
        return _DONE
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    try:
        return obj["choices"][0]["delta"].get("content")
    except (KeyError, IndexError):
        return None
