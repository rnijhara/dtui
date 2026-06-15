"""Simple mode: a streaming chat / coding-agent view.

Talks to any OpenAI-compatible endpoint through a :class:`ChatProvider`. If an
:class:`~dtui.agent.loop.AgentLoop` is supplied, messages run through the tool
loop instead of plain streaming, giving a lightweight coding agent. Streaming
updates land on a per-message widget via the app thread, so the UI stays
responsive while a worker thread drives the provider.
"""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Input, Static
from textual.worker import Worker

from dtui.agent.loop import AgentLoop
from dtui.providers.base import ChatProvider


class ChatView(Widget):
    DEFAULT_CSS = """
    ChatView { height: 1fr; }
    ChatView #transcript { height: 1fr; padding: 1 2; overflow-y: auto; }
    ChatView #prompt { dock: bottom; }
    ChatView .message { width: 100%; height: auto; min-height: 1; }
    ChatView .user { color: $accent; margin-top: 1; }
    ChatView .assistant { margin-top: 1; }
    ChatView .tool { color: #888888; }
    """

    def __init__(
        self,
        provider: ChatProvider | None = None,
        agent: AgentLoop | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.provider = provider
        self.agent = agent
        self._messages: list[dict[str, str]] = []
        self._busy = False

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="transcript")
        yield Input(placeholder="Message  (enter to send)", id="prompt")

    async def _append(self, markup: str, classes: str = "") -> Static:
        widget = Static(markup, classes=f"message {classes}".strip())
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(widget)
        self._scroll_to_end()
        return widget

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self._busy:
            return
        prompt = self.query_one("#prompt", Input)
        prompt.value = ""
        await self._append(f"[b]you[/b]\n{escape(text)}", classes="user")
        if self.provider is None and self.agent is None:
            await self._append("[red]No model configured.[/red]", classes="assistant")
            return
        self._set_busy(True)
        target = await self._append(self._assistant_markup(""), classes="assistant")
        if self.agent is not None:
            self._run_agent(text, target)
        else:
            self._messages.append({"role": "user", "content": text})
            self._run_stream(list(self._messages), target)

    @property
    def messages(self) -> list[dict[str, str]]:
        return [dict(m) for m in self._messages]

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        try:
            prompt = self.query_one("#prompt", Input)
            prompt.disabled = busy
            if not busy and not getattr(self.app, "command_mode", True):
                prompt.focus()
        except Exception:
            pass

    def _scroll_to_end(self) -> None:
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
            self.call_after_refresh(
                transcript.scroll_end,
                animate=False,
                immediate=True,
                force=True,
                y_axis=True,
            )
        except Exception:
            pass

    def _update_assistant(self, target: Static, text: str) -> None:
        target.update(self._assistant_markup(text))
        self._scroll_to_end()

    def _assistant_markup(self, text: str) -> str:
        return f"[grey50]────────────────[/grey50]\n[b]assistant[/b]\n{escape(text)}"

    def _finish_assistant(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})
        self._set_busy(False)
        self._scroll_to_end()

    def _show_error(self, target: Static, e: Exception) -> None:
        target.update(f"[red]{escape(type(e).__name__)}: {escape(str(e))}[/red]")
        self._set_busy(False)
        self._scroll_to_end()

    # -- workers -----------------------------------------------------------

    def _run_stream(self, messages: list[dict[str, str]], target: Static) -> Worker:
        def job() -> None:
            acc = ""
            try:
                for delta in self.provider.stream(messages):  # type: ignore[union-attr]
                    acc += delta
                    self.app.call_from_thread(self._update_assistant, target, acc)
                self.app.call_from_thread(self._finish_assistant, acc)
            except Exception as e:  # surface errors in-line
                self.app.call_from_thread(self._show_error, target, e)

        return self.run_worker(job, thread=True, exclusive=False)

    def _run_agent(self, text: str, target: Static) -> Worker:
        def job() -> None:
            acc = ""
            try:
                for ev in self.agent.run(text):  # type: ignore[union-attr]
                    if ev.kind == "tool_call":
                        acc += f"\n[grey50]· {escape(ev.name)}({escape(str(ev.data))})[/grey50]\n"
                    elif ev.kind == "final":
                        acc += escape(ev.text)
                    self.app.call_from_thread(
                        target.update,
                        f"[grey50]────────────────[/grey50]\n[b]assistant[/b]\n{acc}",
                    )
                    self.app.call_from_thread(self._scroll_to_end)
                self.app.call_from_thread(self._set_busy, False)
            except Exception as e:
                self.app.call_from_thread(self._show_error, target, e)

        return self.run_worker(job, thread=True, exclusive=False)
