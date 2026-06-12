"""The ChatProvider talks to a mocked OpenAI-compatible endpoint."""

import httpx

from dtui.providers.openai_chat import OpenAIChatProvider


class _FakeStream:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_stream_concatenates_deltas(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        "",
        'data: {"choices":[{"delta":{"content":", world"}}]}',
        "data: [DONE]",
        'data: {"choices":[{"delta":{"content":"IGNORED"}}]}',
    ]
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _FakeStream(lines))
    provider = OpenAIChatProvider("http://x/v1", "m", api_key="k")
    out = "".join(provider.stream([{"role": "user", "content": "hi"}]))
    assert out == "Hello, world"


def test_stream_sends_auth_and_stream_flag(monkeypatch):
    captured = {}

    def fake_stream(method, url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeStream(["data: [DONE]"])

    monkeypatch.setattr(httpx, "stream", fake_stream)
    provider = OpenAIChatProvider("http://host/v1", "mymodel", api_key="secret")
    list(provider.stream([{"role": "user", "content": "hi"}]))
    assert captured["url"] == "http://host/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["json"]["stream"] is True
    assert captured["json"]["model"] == "mymodel"


def test_extra_body_passthrough(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        httpx, "stream", lambda *a, **k: captured.update(k) or _FakeStream(["data: [DONE]"])
    )
    provider = OpenAIChatProvider("http://x/v1", "m", extra_body={"diffusing": True})
    list(provider.stream([{"role": "user", "content": "hi"}]))
    assert captured["json"]["diffusing"] is True


def test_complete_returns_tool_calls(monkeypatch):
    data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "1", "function": {"name": "list_dir", "arguments": "{}"}}
                    ],
                }
            }
        ]
    }
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResp(data))
    provider = OpenAIChatProvider("http://x/v1", "m")
    msg = provider.complete(
        [{"role": "user", "content": "list files"}],
        tools=[{"type": "function", "function": {"name": "list_dir"}}],
    )
    assert msg["tool_calls"][0]["function"]["name"] == "list_dir"
