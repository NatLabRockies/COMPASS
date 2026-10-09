"""Tests for compass._cli.search"""

import json
from pathlib import Path

import pytest

import compass._cli.search as cli_module
import compass.web.search as web_search_module
from compass._cli.main import main
from compass.pb import COMPASS_PB


@pytest.fixture(autouse=True)
def reset_compass_progress():
    """Isolate CLI invocations using the shared progress display"""
    COMPASS_PB.reset()
    yield
    COMPASS_PB.reset()
    COMPASS_PB.console = None


@pytest.fixture
def cfg_file(tmp_path):
    """Create a minimal config file for CLI tests"""
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.write_text(
        "County,State\nAdams,Colorado\n", encoding="utf-8"
    )
    fp = tmp_path / "config.json"
    fp.write_text(
        json.dumps(
            {
                "tech": "wind",
                "jurisdiction_fp": "./jurisdictions.csv",
                "out_dir": "./output",
                "simple_se_result_sort": True,
            }
        ),
        encoding="utf-8",
    )
    return fp


@pytest.fixture
def search_backend(monkeypatch):
    """Keep CLI search tests offline at the external backend boundary"""
    calls = []

    async def backend(queries, **kwargs):  # ruff:ignore[unused-async]
        calls.append((queries, kwargs))
        return []

    monkeypatch.setattr(
        web_search_module, "search_with_fallback_with_attrs", backend
    )
    return calls


def test_search_writes_manifest(cli_runner, cfg_file, search_backend):
    """Persist a JSON report through the normal CLI workflow"""
    result = cli_runner.invoke(
        main, ["search", "-c", str(cfg_file), "--no-progress"]
    )

    assert result.exit_code == 0, result.output
    report_fp = cfg_file.parent / "output" / "search_result_manifest.json"
    payload = json.loads(report_fp.read_text(encoding="utf-8"))
    assert payload["tech"] == "wind"
    assert payload["num_jurisdictions_searched"] == 1
    assert len(search_backend) == 1


def test_search_summary_stdout(cli_runner, cfg_file, search_backend):
    """Emit the persisted search report summary when requested"""
    result = cli_runner.invoke(
        main,
        ["search", "-c", str(cfg_file), "--no-progress", "--summarize"],
    )

    assert result.exit_code == 0, result.output
    report_fp = cfg_file.parent / "output" / "search_result_manifest.json"
    payload = json.loads(report_fp.read_text(encoding="utf-8"))
    assert cli_module.summary(payload) in result.output
    assert len(search_backend) == 1


def test_search_n_top_urls_overrides_config(
    cli_runner, cfg_file, search_backend
):
    """Override configured top URL count with a config CLI option"""
    result = cli_runner.invoke(
        main,
        [
            "search",
            "-c",
            str(cfg_file),
            "--no-progress",
            "--num_urls_to_check_per_jurisdiction=12",
        ],
    )

    assert result.exit_code == 0, result.output
    assert search_backend[0][1]["num_urls"] == 12


def test_search_plugin_registers_one_shot(
    cli_runner, cfg_file, monkeypatch, search_backend
):
    """Register one-shot plugin when plugin option is supplied"""
    calls = []

    monkeypatch.setattr(
        cli_module,
        "create_schema_based_one_shot_extraction_plugin",
        lambda **kwargs: calls.append(kwargs),
    )

    result = cli_runner.invoke(
        main,
        [
            "search",
            "-c",
            str(cfg_file),
            "--no-progress",
            "-p",
            "plugin.json5",
        ],
    )

    assert result.exit_code == 0
    assert calls == [{"config": "plugin.json5", "tech": "wind"}]
    assert len(search_backend) == 1


@pytest.mark.parametrize("no_progress", [True, False])
def test_persistent_search_command(
    cli_runner, tmp_path, monkeypatch, no_progress
):
    """Use normal CLI configuration, logs, shards, and request overrides"""
    calls = []

    async def backend(queries, **kwargs):  # ruff:ignore[unused-async]
        calls.append((queries, kwargs))
        return []

    monkeypatch.setattr(
        web_search_module, "search_with_fallback_with_attrs", backend
    )
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.write_text(
        "County,State\nAdams,Colorado\n", encoding="utf-8"
    )
    config_fp = tmp_path / "search.json"
    config_fp.write_text(
        json.dumps(
            {
                "out_dir": "./output",
                "tech": "wind",
                "jurisdiction_fp": "./jurisdictions.csv",
                "simple_se_result_sort": True,
            }
        ),
        encoding="utf-8",
    )
    args = [
        "search",
        "-c",
        str(config_fp),
        "--num_urls_to_check_per_jurisdiction=3",
    ]
    if no_progress:
        args.append("--no-progress")
    result = cli_runner.invoke(main, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    output = tmp_path / "output"
    manifest_fp = output / "search_result_manifest.json"
    manifest = json.loads(manifest_fp.read_text(encoding="utf-8"))
    assert manifest["num_urls_requested"] == 3
    assert manifest["filtered_result_stats"]["total"] == 0
    assert len(calls) == 1
    assert calls[0][1]["num_urls"] == 3
    assert {path.name for path in output.iterdir()} == {
        "logs",
        "se_results",
        "search_result_manifest.json",
    }
    info = manifest["jurisdictions"][0]
    assert info["num_kept_results"] == 0
    assert (output / "logs" / "main.log").is_file()
    assert len(list((output / "logs").glob("*.log"))) >= 2
    conflict = cli_runner.invoke(main, args)
    assert conflict.exit_code != 0
    assert "already exists" in conflict.output
    assert len(calls) == 1


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
