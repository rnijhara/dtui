"""A small, pi-style tool set for the coding agent.

Five tools, real filesystem and shell access, no permission system of its own
(it runs with your permissions, like the reference it is modeled on). Each tool
is a plain function plus an OpenAI tool schema. Keep this list short on purpose.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

Tool = Callable[..., str]


def read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} bytes to {path}"


def edit_file(path: str, old: str, new: str) -> str:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        return f"error: substring not found in {path}"
    if text.count(old) > 1:
        return f"error: substring is not unique in {path} ({text.count(old)} matches)"
    p.write_text(text.replace(old, new), encoding="utf-8")
    return f"edited {path}"


def list_dir(path: str = ".") -> str:
    entries = sorted(Path(path).iterdir(), key=lambda e: (e.is_file(), e.name))
    return "\n".join(("" if e.is_dir() else "  ") + e.name + ("/" if e.is_dir() else "") for e in entries)


def bash(command: str) -> str:
    proc = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=120
    )
    out = proc.stdout + proc.stderr
    return out.strip() or f"(exit {proc.returncode}, no output)"


def _schema(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


_STR = {"type": "string"}

REGISTRY: dict[str, Tool] = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "list_dir": list_dir,
    "bash": bash,
}

SCHEMAS: list[dict] = [
    _schema("read_file", "Read a file and return its contents.", {"path": _STR}, ["path"]),
    _schema("write_file", "Write content to a file, creating parents.", {"path": _STR, "content": _STR}, ["path", "content"]),
    _schema("edit_file", "Replace a unique substring in a file.", {"path": _STR, "old": _STR, "new": _STR}, ["path", "old", "new"]),
    _schema("list_dir", "List entries in a directory.", {"path": _STR}, []),
    _schema("bash", "Run a shell command and return combined stdout/stderr.", {"command": _STR}, ["command"]),
]


def dispatch(name: str, arguments: dict[str, Any]) -> str:
    """Run a tool by name, returning its string result (never raising)."""
    fn = REGISTRY.get(name)
    if fn is None:
        return f"error: unknown tool {name!r}"
    try:
        return fn(**arguments)
    except Exception as e:  # surface tool errors to the model as text
        return f"error: {type(e).__name__}: {e}"
