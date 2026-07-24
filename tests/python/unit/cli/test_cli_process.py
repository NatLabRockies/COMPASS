"""Tests for compass._cli.process"""

from pathlib import Path

import pytest
import click

import compass._cli.process as process_module
from compass._cli.process import process


def test_process_applies_extra_cli_config_overrides(monkeypatch):
    """Process command merges extra CLI options onto the config"""
    captured = {}

    monkeypatch.setattr(
        process_module,
        "load_config",
        lambda *_: {
            "out_dir": "./outputs",
            "tech": "solar",
            "jurisdiction_fp": "./jurisdictions.csv",
            "perform_website_search": True,
            "log_level": "INFO",
        },
    )

    def fake_run_async_command(
        config, request_class, verbose, no_progress, out_dir_exists=None
    ):
        captured["config"] = config
        captured["request_class"] = request_class
        captured["verbose"] = verbose
        captured["no_progress"] = no_progress
        captured["out_dir_exists"] = out_dir_exists

    monkeypatch.setattr(process_module, "run_async_command", fake_run_async_command)

    ctx = click.Context(process)
    ctx.args = [
        "--tech",
        "wind",
        "--max-num-concurrent-browsers=12",
        "--perform-website-search=false",
    ]

    with ctx:
        process.callback("config.json5", 0, False, None, None)

    assert captured["config"]["tech"] == "wind"
    assert captured["config"]["max_num_concurrent_browsers"] == 12
    assert captured["config"]["perform_website_search"] is False


def test_process_rejects_unknown_extra_cli_overrides(tmp_path, cli_runner):
    """Process command fails on unknown extra CLI options"""
    config_path = tmp_path / "config.json5"
    jurisdiction_path = tmp_path / "jurisdictions.csv"
    jurisdiction_path.write_text("County,State\nExample,Colorado\n")
    config_path.write_text(
        """
        {
            out_dir: './outputs',
            tech: 'solar',
            jurisdiction_fp: './jurisdictions.csv',
        }
        """,
        encoding="utf-8",
    )

    result = cli_runner.invoke(
        process,
        [
            "--config",
            str(config_path),
            "--not-a-real-key",
            "value",
        ],
    )

    assert result.exit_code != 0
    assert "Unknown config override key" in result.output


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
