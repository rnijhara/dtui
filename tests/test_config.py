from dtui import config
from dtui.config import ChatConfig


def test_chat_config_reads_local_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "config_file", lambda: tmp_path / "missing.toml")
    monkeypatch.delenv("DTUI_BASE_URL", raising=False)
    monkeypatch.delenv("DTUI_MODEL", raising=False)
    monkeypatch.delenv("DTUI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "DTUI_BASE_URL=https://openrouter.ai/api/v1",
                "DTUI_MODEL=xiaomi/mimo-v2.5-pro",
                "OPENROUTER_KEY=from-dotenv",
            ]
        )
    )

    cfg = ChatConfig.resolve()

    assert cfg.base_url == "https://openrouter.ai/api/v1"
    assert cfg.model == "xiaomi/mimo-v2.5-pro"
    assert cfg.api_key == "from-dotenv"


def test_chat_config_precedence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config_toml = tmp_path / "config.toml"
    config_toml.write_text(
        """
[chat]
base_url = "http://from-config/v1"
model = "from-config"
api_key = "config-key"
"""
    )
    (tmp_path / ".env").write_text(
        """
DTUI_BASE_URL=http://from-dotenv/v1
DTUI_MODEL=from-dotenv
OPENROUTER_KEY=dotenv-key
"""
    )
    monkeypatch.setattr(config, "config_file", lambda: config_toml)
    monkeypatch.setenv("DTUI_MODEL", "from-env")

    cfg = ChatConfig.resolve(base_url="http://from-flag/v1")

    assert cfg.base_url == "http://from-flag/v1"
    assert cfg.model == "from-env"
    assert cfg.api_key == "dotenv-key"
