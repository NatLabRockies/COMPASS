"""Integration tests for COMPASS `search` CLI"""

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


@pytest.mark.parametrize("no_progress", [True, False])
def test_search_continue_reuses_nonempty_shard(
    cli_runner, cfg_file, search_backend, no_progress
):
    """Reuse a successful shard without calling the search provider"""
    args = ["search", "-c", str(cfg_file)]
    if no_progress:
        args.append("--no-progress")
    initial = cli_runner.invoke(main, args)
    assert initial.exit_code == 0, initial.output
    output = cfg_file.parent / "output"
    shard_fp = next((output / "se_results").glob("*.json"))
    shard = json.loads(shard_fp.read_text(encoding="utf-8"))
    shard.update(
        num_results=1,
        num_kept_results=1,
        search_engine_counts={"Google": 1},
        results=[
            {
                "url": "https://example.org/ordinance.pdf",
                "search_engine": "Google",
                "overall_rank": 1,
                "filtered_reason": None,
            }
        ],
    )
    shard_fp.write_text(json.dumps(shard), encoding="utf-8")
    original_bytes = shard_fp.read_bytes()
    COMPASS_PB.reset()

    result = cli_runner.invoke(
        main,
        [
            *args,
            "--out-dir-exists",
            "continue",
            "--target",
            "num_results=1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(search_backend) == 1
    assert shard_fp.read_bytes() == original_bytes
    manifest = json.loads(
        (output / "search_result_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["result_stats"]["total"] == 1


@pytest.mark.parametrize("no_progress", [True, False])
def test_search_continue_selectively_purges_shards(
    cli_runner, tmp_path, monkeypatch, no_progress
):
    """Purge only failing shards and search missing ones exactly once"""
    counties = ["Adams", "Boulder", "Denver", "Jefferson", "Weld"]
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.write_text(
        "County,State\n"
        + "".join(f"{county},Colorado\n" for county in counties),
        encoding="utf-8",
    )
    config_fp = tmp_path / "search.json"
    output = tmp_path / "output"
    config_fp.write_text(
        json.dumps(
            {
                "tech": "wind",
                "jurisdiction_fp": str(jurisdiction_fp),
                "out_dir": str(output),
                "simple_se_result_sort": True,
                "search_engines": [
                    {"se_name": "PlaywrightGoogleLinkSearch"},
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = []
    purged_paths = []
    continuing = False

    async def backend(queries, **kwargs):  # ruff:ignore[unused-async]
        county = next(county for county in counties if county in queries[0])
        calls.append(county)
        if continuing:
            assert not (output / "search_result_manifest.json").exists()
            assert all(
                not path.exists()
                for path in purged_paths
                if county in path.name
            )
            count, engine = 3, "Google"
        else:
            count = {
                "Adams": 3,
                "Boulder": 1,
                "Denver": 3,
                "Jefferson": 0,
                "Weld": 3,
            }[county]
            engine = "Bing" if county == "Denver" else "Google"
        return [
            {
                "url": f"https://example.org/{county}/{continuing}/{index}",
                "search_engine": engine,
                "filtered_reason": None,
                "overall_rank": index + 1,
            }
            for index in range(count)
        ]

    monkeypatch.setattr(
        web_search_module, "search_with_fallback_with_attrs", backend
    )
    args = ["search", "-c", str(config_fp)]
    if no_progress:
        args.append("--no-progress")
    initial = cli_runner.invoke(main, args)
    assert initial.exit_code == 0, initial.output
    shard_paths = {
        county: next((output / "se_results").glob(f"{county}_*.json"))
        for county in counties
    }
    original_bytes = shard_paths["Adams"].read_bytes()
    shard_paths["Weld"].unlink()
    purged_paths = [shard_paths[county] for county in counties[1:]]
    continuing = True
    calls.clear()
    COMPASS_PB.reset()
    result = cli_runner.invoke(
        main,
        [
            *args,
            "--out-dir-exists",
            "continue",
            "--target",
            "num_results=3",
            "--target",
            "search_engine_counts.Google=3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert sorted(calls) == sorted(counties[1:])
    assert shard_paths["Adams"].read_bytes() == original_bytes
    manifest = json.loads((output / "search_result_manifest.json").read_text())
    assert manifest["num_jurisdictions_searched"] == 5
    assert manifest["result_stats"]["total"] == 15
    for county in counties[1:]:
        record = json.loads(shard_paths[county].read_text())
        assert all("/True/" in row["url"] for row in record["results"])


@pytest.mark.parametrize("no_progress", [True, False])
@pytest.mark.parametrize("minimum", [0, 3])
def test_search_continue_saves_unmet_targets(
    cli_runner, cfg_file, search_backend, no_progress, minimum
):
    """Save failing shards before exiting nonzero without a second pass"""
    args = ["search", "-c", str(cfg_file)]
    if no_progress:
        args.append("--no-progress")
    initial = cli_runner.invoke(main, args)
    assert initial.exit_code == 0, initial.output
    COMPASS_PB.reset()
    output = cfg_file.parent / "output"
    result = cli_runner.invoke(
        main,
        [
            *args,
            "--out-dir-exists",
            "continue",
            "--target",
            f"num_results={minimum}",
            "--summarize",
        ],
    )

    assert result.exit_code == 1, result.output
    assert len(search_backend) == 2
    assert "Adams County, Colorado" in result.output
    assert "COMPASS search-only summary" in result.output
    assert COMPASS_PB.console is None
    manifest = json.loads((output / "search_result_manifest.json").read_text())
    failures = ["Adams County, Colorado: results=0 (minimum 1)"]
    if minimum:
        failures.append(
            f"Adams County, Colorado: num_results=0 (minimum {minimum})"
        )
    assert manifest["failed_targets"] == failures
    assert f"Error: Failed {len(failures)} targets" in result.output
    assert manifest["result_stats"]["total"] == 0
    assert len(list((output / "se_results").glob("*.json"))) == 1


@pytest.mark.parametrize(
    "option, error",
    [
        (
            ["--target", "num_results=-1"],
            "requires a nonnegative integer",
        ),
        (
            ["--target", "num_results=1.5"],
            "requires a nonnegative integer",
        ),
        (
            ["--target", "result_stats.min=3"],
            "Unknown search shard target 'result_stats.min'",
        ),
        (
            ["--target", "search_engine_counts.Unknown=3"],
            "Unknown search shard target 'search_engine_counts.Unknown'",
        ),
        (
            ["--target", "num_kept_results=6"],
            "exceeds the configured retained URL limit (5)",
        ),
        (
            ["--out-dir-exists", "overwrite"],
            "conflicts with --out-dir-exists",
        ),
        (["--target", "num_results"], "use METRIC=MINIMUM"),
    ],
)
def test_invalid_continue_options_leave_outputs_untouched(
    cli_runner, cfg_file, search_backend, option, error
):
    """Validate requirements and conflicting policies before purging"""
    args = ["search", "-c", str(cfg_file), "--no-progress"]
    initial = cli_runner.invoke(main, args)
    assert initial.exit_code == 0, initial.output
    output = cfg_file.parent / "output"
    before = {path: path.read_bytes() for path in output.rglob("*.json")}
    COMPASS_PB.reset()
    result = cli_runner.invoke(
        main,
        [
            *args,
            "--out-dir-exists",
            "continue",
            "--target",
            "num_kept_results=0",
            *option,
        ],
    )
    assert result.exit_code == 1, result.output
    assert error in f"{result.output}\n{result.exception}"
    assert len(search_backend) == 1
    assert {
        path: path.read_bytes() for path in output.rglob("*.json")
    } == before


def test_cli_targets_keep_strictest_repeated_minimum(
    cli_runner, cfg_file, search_backend
):
    """Use the strictest repeated target without dropping other metrics"""
    result = cli_runner.invoke(
        main,
        [
            "search",
            "-c",
            str(cfg_file),
            "--no-progress",
            "--target",
            "num_results=3",
            "--target",
            "num_results=1",
            "--target",
            "num_kept_results=2",
        ],
    )
    assert result.exit_code == 1, result.output
    assert len(search_backend) == 1
    manifest = json.loads(
        (
            cfg_file.parent / "output" / "search_result_manifest.json"
        ).read_text()
    )
    assert manifest["failed_targets"] == [
        "Adams County, Colorado: results=0 (minimum 1)",
        "Adams County, Colorado: num_results=0 (minimum 3)",
        "Adams County, Colorado: num_kept_results=0 (minimum 2)",
    ]


def test_continue_searches_without_any_shard_directory(
    cli_runner, cfg_file, search_backend
):
    """Search every missing jurisdiction even without a shard directory"""
    output = cfg_file.parent / "empty_output"
    output.mkdir()
    result = cli_runner.invoke(
        main,
        [
            "search",
            "-c",
            str(cfg_file),
            f"--out_dir={output}",
            "--out-dir-exists",
            "continue",
            "--target",
            "num_results=1",
            "--no-progress",
        ],
    )
    assert result.exit_code == 1, result.output
    assert len(search_backend) == 1
    assert len(list((output / "se_results").glob("*.json"))) == 1
    assert (output / "search_result_manifest.json").exists()
    assert not (cfg_file.parent / "output").exists()


def test_fresh_search_targets_are_checked_after_saving(
    cli_runner, cfg_file, search_backend
):
    """Use the same minimum evaluator for a fresh single-pass search"""
    result = cli_runner.invoke(
        main,
        [
            "search",
            "-c",
            str(cfg_file),
            "--target",
            "num_results=3",
            "--no-progress",
        ],
    )
    assert result.exit_code == 1, result.output
    assert len(search_backend) == 1
    assert (
        cfg_file.parent / "output" / "search_result_manifest.json"
    ).exists()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
