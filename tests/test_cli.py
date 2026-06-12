from click.testing import CliRunner

from dtui import __version__
from dtui.cli import main


def test_help_lists_modes():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "chat" in result.output
    assert "viz" in result.output


def test_version():
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_viz_help():
    result = CliRunner().invoke(main, ["viz", "--help"])
    assert result.exit_code == 0
    assert "trajectory" in result.output.lower()
