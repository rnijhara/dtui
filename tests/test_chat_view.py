"""Pilot tests for chat transcript behavior."""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Input, Static

from dtui.tui.app import DtuiApp
from dtui.tui.chat import ChatView


class ScriptedProvider:
    name = "scripted"

    def __init__(self, responses: list[list[str]]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []

    def stream(self, messages):
        self.calls.append([dict(m) for m in messages])
        response = self.responses[len(self.calls) - 1]
        yield from response


async def _submit(app: DtuiApp, pilot, text: str) -> None:
    if app.command_mode:
        await pilot.press("i")
        await pilot.pause()
    prompt = app.query_one("#prompt", Input)
    prompt.value = text
    await pilot.press("enter")


async def _wait_done(chat: ChatView, pilot, provider: ScriptedProvider, calls: int) -> None:
    for _ in range(200):
        await pilot.pause()
        if len(provider.calls) >= calls and not chat._busy:
            return
    raise AssertionError("chat stream did not finish")


def _transcript_items(app: DtuiApp) -> list[str]:
    transcript = app.query_one("#transcript", VerticalScroll)
    return [str(child.content) for child in transcript.children if isinstance(child, Static)]


async def test_chat_appends_turns_and_sends_history():
    provider = ScriptedProvider([["Hello"], ["I remember that."]])
    app = DtuiApp(chat_provider=provider, start_mode="chat")
    async with app.run_test() as pilot:
        chat = app.query_one("#chat", ChatView)

        await _submit(app, pilot, "first")
        await _wait_done(chat, pilot, provider, calls=1)

        await _submit(app, pilot, "second")
        await _wait_done(chat, pilot, provider, calls=2)

        assert _transcript_items(app) == [
            "[b]you[/b]\nfirst",
            "[grey50]────────────────[/grey50]\n[b]assistant[/b]\nHello",
            "[b]you[/b]\nsecond",
            "[grey50]────────────────[/grey50]\n[b]assistant[/b]\nI remember that.",
        ]
        assert provider.calls[0] == [{"role": "user", "content": "first"}]
        assert provider.calls[1] == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "second"},
        ]
        assert chat.messages == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "I remember that."},
        ]


async def test_chat_keeps_full_long_streamed_response_visible_in_transcript():
    chunks = [f"line {i}\n" for i in range(120)]
    provider = ScriptedProvider([chunks])
    app = DtuiApp(chat_provider=provider, start_mode="chat")
    async with app.run_test(size=(80, 20)) as pilot:
        chat = app.query_one("#chat", ChatView)

        await _submit(app, pilot, "write a long answer")
        await _wait_done(chat, pilot, provider, calls=1)

        assistant_item = _transcript_items(app)[1]
        assert assistant_item.startswith("[grey50]────────────────[/grey50]\n[b]assistant[/b]\n")
        assert "line 0" in assistant_item
        assert "line 119" in assistant_item
        assert assistant_item.count("line ") == 120

        transcript = app.query_one("#transcript", VerticalScroll)
        assert transcript.scroll_y == transcript.max_scroll_y
