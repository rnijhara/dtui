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
    ChatView #transcript { height: 1fr; padding: 1 2; }
    ChatView #prompt { dock: bottom; }
    ChatView .user { color: $accent; margin-top: 1; }
    ChatView .assistant { margin-top: 0; }
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

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="transcript")
        yield Input(placeholder="Message  (enter to send)", id="prompt")

    async def _append(self, markup: str, classes: str = "") -> Static:
        widget = Static(markup, classes=classes)
        await self.query_one("#transcript", VerticalScroll).mount(widget)
        return widget

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        self.query_one("#prompt", Input).value = ""
        await self._append(f"[b]you[/b]  {escape(text)}", classes="user")
        if self.provider is None and self.agent is None:
            await self._append("[red]No model configured.[/red]", classes="assistant")
            return
        target = await self._append("", classes="assistant")
        if self.agent is not None:
            self._run_agent(text, target)
        else:
            self._run_stream(text, target)

    # -- workers -----------------------------------------------------------

    def _run_stream(self, text: str, target: Static) -> Worker:
        def job() -> None:
            acc = ""
            messages = [{"role": "user", "content": text}]
            try:
                for delta in self.provider.stream(messages):  # type: ignore[union-attr]
                    acc += delta
                    self.app.call_from_thread(target.update, escape(acc))
            except Exception as e:  # surface errors in-line
                self.app.call_from_thread(
                    target.update, f"[red]{escape(type(e).__name__)}: {escape(str(e))}[/red]"
                )

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
                    self.app.call_from_thread(target.update, acc)
            except Exception as e:
                self.app.call_from_thread(
                    target.update, f"[red]{escape(type(e).__name__)}: {escape(str(e))}[/red]"
                )

        return self.run_worker(job, thread=True, exclusive=False)
