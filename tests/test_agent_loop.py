"""The agent loop runs a scripted tool call against a fake provider."""

import json

from dtui.agent.loop import AgentLoop


class FakeProvider:
    """Returns a scripted sequence of completion messages."""

    def __init__(self, scripted):
        self.scripted = scripted
        self.calls = 0

    def complete(self, messages, tools=None):
        msg = self.scripted[self.calls]
        self.calls += 1
        return msg


def test_agent_executes_tool_then_answers(tmp_path):
    target = tmp_path / "note.txt"
    target.write_text("the answer is 42")

    scripted = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": str(target)}),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "The note says the answer is 42."},
    ]

    loop = AgentLoop(FakeProvider(scripted))
    events = list(loop.run("what does note.txt say?"))

    kinds = [e.kind for e in events]
    assert "tool_call" in kinds
    assert kinds[-1] == "final"

    tool_result = next(e for e in events if e.kind == "tool_result")
    assert "the answer is 42" in tool_result.text
    assert events[-1].text == "The note says the answer is 42."


def test_agent_stops_at_max_steps():
    # Provider always asks for another tool call; loop must terminate.
    always_tool = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "x", "function": {"name": "list_dir", "arguments": "{}"}}],
    }

    class Loop(FakeProvider):
        def complete(self, messages, tools=None):
            return always_tool

    loop = AgentLoop(Loop([]), max_steps=3)
    events = list(loop.run("loop forever"))
    assert events[-1].kind == "final"
    assert "max tool steps" in events[-1].text
