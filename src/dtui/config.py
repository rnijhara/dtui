"""Lightweight config resolution: TOML file plus environment overrides.

Precedence (low to high): config file -> environment -> explicit CLI flags.
Only the standard library is used (``tomllib`` is stdlib on 3.11+).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_config_path

APP_NAME = "dtui"


def config_file() -> Path:
    return user_config_path(APP_NAME) / "config.toml"


@dataclass
class ChatConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "local"
    api_key: str | None = None

    @classmethod
    def resolve(
        cls,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> "ChatConfig":
        data: dict = {}
        path = config_file()
        if path.exists():
            with path.open("rb") as f:
                data = tomllib.load(f).get("chat", {})
        return cls(
            base_url=base_url or os.environ.get("DTUI_BASE_URL") or data.get("base_url", cls.base_url),
            model=model or os.environ.get("DTUI_MODEL") or data.get("model", cls.model),
            api_key=api_key or os.environ.get("DTUI_API_KEY") or data.get("api_key"),
        )
