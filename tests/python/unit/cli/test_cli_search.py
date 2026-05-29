"""Tests for compass._cli.search"""

import asyncio
import json
from pathlib import Path

import pytest

import compass._cli.search as cli_module


@pytest.fixture
def cfg_file(tmp_path):
    """Create a minimal config file for CLI tests"""
    fp = tmp_path / "config.json"
    fp.write_text("{}", encoding="utf-8")
    return fp


def test_search_json_stdout(cli_runner, cfg_file, monkeypatch):
    """Emit JSON report to stdout by default"""

    def _write_json_stdout(report, out_path=None):
        _ = out_path
        print(json.dumps(report))

    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda *_: {"tech": "wind", "jurisdiction_fp": "dummy.csv"},
    )
    monkeypatch.setattr(cli_module, "setup_cli_logging", lambda *_, **__: None)
    monkeypatch.setattr(
        cli_module,
        "run_search",
        _async_returns({"tech": "wind", "jurisdictions": []}),
    )
    monkeypatch.setattr(
        cli_module,
        "write_search_report",
        _write_json_stdout,
    )

    result = cli_runner.invoke(
        cli_module.search,
        ["-c", str(cfg_file)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["tech"] == "wind"


def test_search_summary_stdout(cli_runner, cfg_file, monkeypatch):
    """Emit summary report to stdout when requested"""

    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda *_: {"tech": "wind", "jurisdiction_fp": "dummy.csv"},
    )
    monkeypatch.setattr(cli_module, "setup_cli_logging", lambda *_, **__: None)
    monkeypatch.setattr(
        cli_module,
        "run_search",
        _async_returns({"tech": "wind", "jurisdictions": []}),
    )
    monkeypatch.setattr(
        cli_module,
        "summary",
        lambda *_: "summary report",
    )

    result = cli_runner.invoke(
        cli_module.search,
        ["-c", str(cfg_file), "--output-format", "summary"],
    )

    assert result.exit_code == 0
    assert result.output.strip() == "summary report"


def test_search_summary_file_output(
    cli_runner, cfg_file, monkeypatch, tmp_path
):
    """Write summary report to output file"""
    out_fp = tmp_path / "summary.txt"

    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda *_: {"tech": "wind", "jurisdiction_fp": "dummy.csv"},
    )
    monkeypatch.setattr(cli_module, "setup_cli_logging", lambda *_, **__: None)
    monkeypatch.setattr(
        cli_module,
        "run_search",
        _async_returns({"tech": "wind", "jurisdictions": []}),
    )
    monkeypatch.setattr(
        cli_module,
        "summary",
        lambda *_: "summary report",
    )

    result = cli_runner.invoke(
        cli_module.search,
        [
            "-c",
            str(cfg_file),
            "--output-format",
            "summary",
            "-o",
            str(out_fp),
        ],
    )

    assert result.exit_code == 0
    assert out_fp.read_text(encoding="utf-8") == "summary report\n"


def test_search_n_top_urls_overrides_config(cli_runner, cfg_file, monkeypatch):
    """Override configured top URL count with CLI option"""
    captured = {}

    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda *_: {
            "tech": "wind",
            "jurisdiction_fp": "dummy.csv",
            "num_urls_to_check_per_jurisdiction": 5,
        },
    )
    monkeypatch.setattr(cli_module, "setup_cli_logging", lambda *_, **__: None)
    monkeypatch.setattr(
        cli_module,
        "run_search",
        _capture_async_kwargs(captured),
    )
    monkeypatch.setattr(
        cli_module,
        "write_search_report",
        lambda *_args, **_kwargs: None,
    )

    result = cli_runner.invoke(
        cli_module.search,
        ["-c", str(cfg_file), "-n", "12"],
    )

    assert result.exit_code == 0
    assert captured["num_urls_to_check_per_jurisdiction"] == 12


def test_search_plugin_registers_one_shot(cli_runner, cfg_file, monkeypatch):
    """Register one-shot plugin when plugin option is supplied"""
    calls = []

    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda *_: {"tech": "wind", "jurisdiction_fp": "dummy.csv"},
    )
    monkeypatch.setattr(cli_module, "setup_cli_logging", lambda *_, **__: None)
    monkeypatch.setattr(
        cli_module,
        "run_search",
        _async_returns({"tech": "wind", "jurisdictions": []}),
    )
    monkeypatch.setattr(
        cli_module,
        "write_search_report",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        cli_module,
        "create_schema_based_one_shot_extraction_plugin",
        lambda **kwargs: calls.append(kwargs),
    )

    result = cli_runner.invoke(
        cli_module.search,
        ["-c", str(cfg_file), "-p", "plugin.json5"],
    )

    assert result.exit_code == 0
    assert calls == [{"config": "plugin.json5", "tech": "wind"}]


def _async_returns(value):
    async def _inner(*_args, **_kwargs):
        await asyncio.sleep(0)
        return value

    return _inner


def _capture_async_kwargs(out_dict):
    async def _inner(*_args, **kwargs):
        await asyncio.sleep(0)
        out_dict.update(kwargs)
        return {"tech": "wind", "jurisdictions": []}

    return _inner


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
