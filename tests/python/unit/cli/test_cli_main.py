"""Tests for COMPASS CLI command registration"""

from pathlib import Path

import pytest

from compass._cli.main import main


@pytest.mark.parametrize(
    "command_name",
    ["collect", "extract", "process", "finalize"],
)
def test_main_help_lists_expected_commands(command_name, cli_runner):
    """Ensure the main CLI exposes the expected subcommands"""
    result = cli_runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert command_name in result.output


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
