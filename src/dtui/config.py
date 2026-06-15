"""Lightweight config resolution: TOML, .env, and environment overrides.

Precedence (low to high): config file -> local .env -> environment -> explicit
CLI flags. Only the standard library is used (``tomllib`` is stdlib on 3.11+).
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


def dotenv_file() -> Path:
    return Path.cwd() / ".env"


def _read_dotenv(path: Path | None = None) -> dict[str, str]:
    path = path or dotenv_file()
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in ("'", '"')
            ):
                value = value[1:-1]
            if key:
                values[key] = value
    return values


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
        dotenv = _read_dotenv()
        return cls(
            base_url=base_url
            or os.environ.get("DTUI_BASE_URL")
            or dotenv.get("DTUI_BASE_URL")
            or data.get("base_url", cls.base_url),
            model=model
            or os.environ.get("DTUI_MODEL")
            or dotenv.get("DTUI_MODEL")
            or data.get("model", cls.model),
            api_key=api_key
            or os.environ.get("DTUI_API_KEY")
            or os.environ.get("OPENROUTER_KEY")
            or dotenv.get("DTUI_API_KEY")
            or dotenv.get("OPENROUTER_KEY")
            or data.get("api_key"),
        )
