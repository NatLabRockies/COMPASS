"""Test COMPASS finalize utilities"""

import os
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from compass.utilities import finalize


def _manifest_for_dirs(dirs, monkeypatch):
    """Run save_run_meta and return serialized manifest"""

    monkeypatch.setattr(finalize.getpass, "getuser", lambda: "testuser")
    start = datetime(2025, 1, 1, 12, 0, 0)
    end = start + timedelta(seconds=1)

    finalize.save_run_meta(
        dirs,
        "wind",
        start,
        end,
        num_jurisdictions_searched=1,
        num_jurisdictions_found=1,
        total_cost=None,
        models={},
    )

    meta = json.loads((dirs.out / "meta.json").read_text(encoding="utf-8"))
    return meta["manifest"]


class DummyModelConfig:
    """Lightweight LLM model config for grouping tests"""

    def __init__(
        self,
        *,
        name,
        llm_call_kwargs=None,
        llm_service_rate_limit,
        text_splitter_chunk_size,
        text_splitter_chunk_overlap,
        client_type,
    ):
        self.name = name
        self.llm_call_kwargs = llm_call_kwargs or {}
        self.llm_service_rate_limit = llm_service_rate_limit
        self.text_splitter_chunk_size = text_splitter_chunk_size
        self.text_splitter_chunk_overlap = text_splitter_chunk_overlap
        self.client_type = client_type

    def __hash__(self):
        return id(self)


def test_save_run_meta_writes_meta_file(tmp_path, monkeypatch):
    """Save run metadata with populated manifest entries"""

    logs = tmp_path / "logs"
    clean_files = tmp_path / "clean"
    jurisdiction_dbs = tmp_path / "jurisdictions"
    ordinance_files = tmp_path / "ordinances"
    for path in (logs, clean_files, jurisdiction_dbs, ordinance_files):
        path.mkdir()

    (tmp_path / "usage.json").write_text("{}", encoding="utf-8")
    (tmp_path / "jurisdictions.json").write_text("{}", encoding="utf-8")
    (tmp_path / "quantitative_ordinances.csv").write_text(
        "header\n",
        encoding="utf-8",
    )

    dirs = SimpleNamespace(
        logs=logs,
        clean_files=clean_files,
        jurisdiction_dbs=jurisdiction_dbs,
        ordinance_files=ordinance_files,
        out=tmp_path,
    )

    monkeypatch.setattr(finalize.getpass, "getuser", lambda: "testuser")

    model = DummyModelConfig(
        name="gpt",
        llm_call_kwargs={"temperature": 0},
        llm_service_rate_limit=5,
        text_splitter_chunk_size=1000,
        text_splitter_chunk_overlap=200,
        client_type="openai",
    )

    start = datetime(2025, 1, 1, 12, 0, 0)
    end = start + timedelta(hours=1, minutes=2, seconds=3)

    seconds = finalize.save_run_meta(
        dirs,
        "wind",
        start,
        end,
        num_jurisdictions_searched=4,
        num_jurisdictions_found=2,
        total_cost=12.34,
        models={"task": model},
    )

    assert seconds == 3723

    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["username"] == "testuser"
    assert meta["technology"] == "wind"
    assert meta["total_time"] == 3723
    assert meta["total_time_string"] == str(end - start)
    assert meta["cost"] == 12.34  # noqa

    manifest = meta["manifest"]
    assert manifest["LOG_DIR"] == "logs"
    assert manifest["CLEAN_FILE_DIR"] == "clean"
    assert manifest["JURISDICTION_DBS_DIR"] == "jurisdictions"
    assert manifest["ORDINANCE_FILES_DIR"] == "ordinances"
    assert manifest["USAGE_FILE"] == "usage.json"
    assert manifest["JURISDICTION_FILE"] == "jurisdictions.json"
    assert manifest["QUAL_DATA_FILE"] == "quantitative_ordinances.csv"
    assert manifest["META_FILE"] == "meta.json"

    model_info = meta["models"][0]
    assert model_info["name"] == "gpt"
    assert model_info["tasks"] == ["task"]
    assert model_info["llm_call_kwargs"] == {"temperature": 0}


def test_save_run_meta_handles_getuser_error(tmp_path, monkeypatch):
    """Fallback to unknown username when getpass fails"""

    def _raise_os_error():
        raise OSError("unavailable")

    monkeypatch.setattr(finalize.getpass, "getuser", _raise_os_error)

    dirs = SimpleNamespace(
        logs=tmp_path / "missing_logs",
        clean_files=tmp_path / "missing_clean",
        jurisdiction_dbs=tmp_path / "missing_jurisdictions",
        ordinance_files=tmp_path / "missing_ordinances",
        out=tmp_path,
    )

    start = datetime(2025, 1, 1, 0, 0, 0)
    end = start + timedelta(days=1, seconds=42)

    seconds = finalize.save_run_meta(
        dirs,
        "solar",
        start,
        end,
        num_jurisdictions_searched=0,
        num_jurisdictions_found=0,
        total_cost=0,
        models={},
    )

    assert seconds == 42 + 60 * 60 * 24

    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["username"] == "Unknown"
    assert meta["cost"] is None
    assert meta["manifest"]["LOG_DIR"] is None
    assert meta["models"] == []


def test_save_run_meta_manifest_walks_up_for_sibling_dirs(
    tmp_path, monkeypatch
):
    """Manifest paths include walk-up segments for sibling directories"""

    run_root = tmp_path / "run"
    out_dir = run_root / "outputs"
    logs = run_root / "logs"
    clean_files = run_root / "clean"
    jurisdiction_dbs = run_root / "jurisdiction_dbs"
    ordinance_files = run_root / "ordinance_files"

    for path in (
        out_dir,
        logs,
        clean_files,
        jurisdiction_dbs,
        ordinance_files,
    ):
        path.mkdir(parents=True, exist_ok=True)

    dirs = SimpleNamespace(
        logs=logs,
        clean_files=clean_files,
        jurisdiction_dbs=jurisdiction_dbs,
        ordinance_files=ordinance_files,
        out=out_dir,
    )

    manifest = _manifest_for_dirs(dirs, monkeypatch)
    assert manifest["LOG_DIR"] == os.path.relpath(logs, out_dir)
    assert manifest["CLEAN_FILE_DIR"] == os.path.relpath(clean_files, out_dir)
    assert manifest["JURISDICTION_DBS_DIR"] == os.path.relpath(
        jurisdiction_dbs, out_dir
    )
    assert manifest["ORDINANCE_FILES_DIR"] == os.path.relpath(
        ordinance_files, out_dir
    )


def test_save_run_meta_manifest_walks_up_two_levels(tmp_path, monkeypatch):
    """Manifest handles directories two levels above output directory"""

    run_root = tmp_path / "run"
    out_dir = run_root / "outputs" / "current"
    logs = run_root / "logs"
    clean_files = out_dir / "clean"
    jurisdiction_dbs = out_dir / "dbs"
    ordinance_files = out_dir / "ordinances"

    for path in (
        out_dir,
        logs,
        clean_files,
        jurisdiction_dbs,
        ordinance_files,
    ):
        path.mkdir(parents=True, exist_ok=True)

    dirs = SimpleNamespace(
        logs=logs,
        clean_files=clean_files,
        jurisdiction_dbs=jurisdiction_dbs,
        ordinance_files=ordinance_files,
        out=out_dir,
    )

    manifest = _manifest_for_dirs(dirs, monkeypatch)
    assert manifest["LOG_DIR"] == os.path.relpath(logs, out_dir)


def test_save_run_meta_manifest_handles_nested_output_paths(
    tmp_path, monkeypatch
):
    """Manifest uses in-tree relative paths for nested directories"""

    out_dir = tmp_path / "run" / "outputs"
    logs = out_dir / "level_one" / "level_two"
    clean_files = out_dir / "clean"
    jurisdiction_dbs = out_dir / "dbs"
    ordinance_files = out_dir / "ordinances"

    for path in (
        out_dir,
        logs,
        clean_files,
        jurisdiction_dbs,
        ordinance_files,
    ):
        path.mkdir(parents=True, exist_ok=True)

    dirs = SimpleNamespace(
        logs=logs,
        clean_files=clean_files,
        jurisdiction_dbs=jurisdiction_dbs,
        ordinance_files=ordinance_files,
        out=out_dir,
    )

    manifest = _manifest_for_dirs(dirs, monkeypatch)
    assert manifest["LOG_DIR"] == os.path.relpath(logs, out_dir)


def test_save_run_meta_manifest_handles_external_directory(
    tmp_path, monkeypatch
):
    """Manifest can serialize paths outside output tree"""

    out_dir = tmp_path / "run" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="compass-external-") as ext_root:
        external_logs = Path(ext_root) / "logs"
        external_logs.mkdir()
        clean_files = out_dir / "clean"
        jurisdiction_dbs = out_dir / "dbs"
        ordinance_files = out_dir / "ordinances"
        for path in (clean_files, jurisdiction_dbs, ordinance_files):
            path.mkdir(parents=True, exist_ok=True)

        dirs = SimpleNamespace(
            logs=external_logs,
            clean_files=clean_files,
            jurisdiction_dbs=jurisdiction_dbs,
            ordinance_files=ordinance_files,
            out=out_dir,
        )

        manifest = _manifest_for_dirs(dirs, monkeypatch)
        assert manifest["LOG_DIR"] == os.path.relpath(external_logs, out_dir)


def test_save_run_meta_manifest_handles_root_only_shared_component(
    monkeypatch,
):
    """Manifest handles paths that share only the filesystem root"""

    roots = [Path("/tmp"), Path("/var/tmp")]  # noqa
    if any(not root.exists() for root in roots):
        pytest.skip("Required temp roots are not available on this system")

    if any(not os.access(root, os.W_OK | os.X_OK) for root in roots):
        pytest.skip("Required temp roots are not writable/executable")

    with (
        tempfile.TemporaryDirectory(prefix="compass-out-", dir="/tmp") as o,
        tempfile.TemporaryDirectory(
            prefix="compass-ext-", dir="/var/tmp"
        ) as e,
    ):
        out_dir = Path(o) / "outputs"
        external_logs = Path(e) / "logs"
        clean_files = out_dir / "clean"
        jurisdiction_dbs = out_dir / "dbs"
        ordinance_files = out_dir / "ordinances"

        for path in (
            out_dir,
            external_logs,
            clean_files,
            jurisdiction_dbs,
            ordinance_files,
        ):
            path.mkdir(parents=True, exist_ok=True)

        out_dir_resolved = out_dir.resolve()
        external_logs_resolved = external_logs.resolve()
        common = os.path.commonpath(
            [str(out_dir_resolved), str(external_logs_resolved)]
        )
        if common != os.path.sep:
            pytest.skip("Could not create root-only shared path topology")

        dirs = SimpleNamespace(
            logs=external_logs,
            clean_files=clean_files,
            jurisdiction_dbs=jurisdiction_dbs,
            ordinance_files=ordinance_files,
            out=out_dir,
        )

        manifest = _manifest_for_dirs(dirs, monkeypatch)
        assert manifest["LOG_DIR"] == os.path.relpath(external_logs, out_dir)


def test_save_run_meta_manifest_handles_out_and_parent(tmp_path, monkeypatch):
    """Manifest supports output directory and its parent as targets"""

    out_dir = tmp_path / "run" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs = SimpleNamespace(
        logs=out_dir,
        clean_files=out_dir.parent,
        jurisdiction_dbs=out_dir / "dbs",
        ordinance_files=out_dir / "ordinances",
        out=out_dir,
    )
    dirs.jurisdiction_dbs.mkdir(parents=True, exist_ok=True)
    dirs.ordinance_files.mkdir(parents=True, exist_ok=True)

    manifest = _manifest_for_dirs(dirs, monkeypatch)
    assert manifest["LOG_DIR"] == "."
    assert manifest["CLEAN_FILE_DIR"] == ".."


def test_doc_infos_to_db_empty_input():
    """No documents returns empty DataFrame"""

    db, count = finalize.doc_infos_to_db([])
    assert count == 0
    assert db.empty
    assert list(db.columns) == finalize._PARSED_COLS


def test_doc_infos_to_db_compiles_and_formats(tmp_path):
    """Compile document info into formatted DataFrame"""

    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame(columns=["feature", "summary"]).to_csv(empty_csv, index=False)

    valid_csv = tmp_path / "valid.csv"
    pd.DataFrame(
        [
            {
                "feature": "Height Limit",
                "summary": "Maximum 100 ft",
                "value": 100,
                "units": "ft",
                "adder": 300,
                "source": "http://example.com/valid",
                "year": 2022,
            }
        ]
    ).to_csv(valid_csv, index=False)

    jurisdiction = SimpleNamespace(
        code="12345",
        county="Example",
        state="EX",
        subdivision_name="Example Township",
        type="county",
    )

    doc_infos = [
        None,
        {"ord_db_fp": None},
        {
            "ord_db_fp": empty_csv,
            "source": "http://example.com/empty",
            "date": (2020, 1, 1),
            "jurisdiction": jurisdiction,
        },
        {
            "ord_db_fp": valid_csv,
            "jurisdiction": jurisdiction,
        },
    ]

    db, count = finalize.doc_infos_to_db(doc_infos)
    assert count == 1
    assert len(db) == 1

    row = db.iloc[0]
    assert row["source"] == "http://example.com/valid"
    assert row["year"] == 2022
    assert row["FIPS"] == "12345"
    assert bool(row["quantitative"]) is True
    assert pd.isna(row["adder"])


def test_save_db_writes_csvs(tmp_path):
    """Split qualitative and quantitative outputs"""

    row_true = dict.fromkeys(finalize._PARSED_COLS)
    row_true.update(
        {
            "county": "County A",
            "state": "ST",
            "subdivision": "Subdivision",
            "jurisdiction_type": "county",
            "FIPS": "00001",
            "feature": "Height",
            "value": 100,
            "units": "ft",
            "summary": "Maximum height",
            "ord_text": "Buildings shall not exceed 100 ft.",
            "year": 2020,
            "source": "http://source",
            "quantitative": True,
        }
    )

    row_false = row_true.copy()
    row_false.update(
        {
            "feature": "Setback",
            "summary": "Setback distance",
            "ord_text": "Turbines shall be set back from property lines.",
            "quantitative": False,
        }
    )

    df = pd.DataFrame([row_true, row_false])
    finalize.save_db(df, tmp_path)

    quant_path = tmp_path / "quantitative_ordinances.csv"
    qual_path = tmp_path / "qualitative_ordinances.csv"
    assert quant_path.exists()
    assert qual_path.exists()

    quant = pd.read_csv(quant_path)
    qual = pd.read_csv(qual_path)
    assert list(quant.columns) == finalize.QUANT_OUT_COLS
    assert len(quant) == 1
    assert list(qual.columns) == finalize.QUAL_OUT_COLS
    assert len(qual) == 1
    assert quant.iloc[0]["feature"] == "Height"
    assert qual.iloc[0]["feature"] == "Setback"
    assert quant.iloc[0]["ord_text"] == "Buildings shall not exceed 100 ft."
    assert (
        qual.iloc[0]["ord_text"]
        == "Turbines shall be set back from property lines."
    )


def test_save_db_with_empty_df(tmp_path):
    """Do nothing when DataFrame is empty"""

    empty_df = pd.DataFrame(columns=finalize._PARSED_COLS)
    finalize.save_db(empty_df, tmp_path)
    assert not (tmp_path / "qualitative_ordinances.csv").exists()
    assert not (tmp_path / "quantitative_ordinances.csv").exists()


def test_db_results_populates_jurisdiction_fields():
    """Populate DataFrame fields from jurisdiction metadata"""

    base_df = pd.DataFrame(
        [
            {
                "feature": "Height",
                "summary": "Max height",
            }
        ]
    )
    jurisdiction = SimpleNamespace(
        code="54321",
        county="County B",
        state="SB",
        subdivision_name="Subdivision B",
        type="city",
    )

    result = finalize._db_results(base_df.copy(), jurisdiction)
    row = result.iloc[0]
    assert row["FIPS"] == "54321"
    assert row["county"] == "County B"
    assert row["jurisdiction_type"] == "city"


def test_empirical_adjustments_caps_adder():
    """Clamp adder values above empirical limit"""

    db = pd.DataFrame({"adder": [300, 150]})
    adjusted = finalize._empirical_adjustments(db.copy())
    assert pd.isna(adjusted.loc[0, "adder"])
    assert adjusted.loc[1, "adder"] == 150

    no_adder_df = pd.DataFrame({"feature": ["Height"]})
    adjusted_no_adder = finalize._empirical_adjustments(no_adder_df.copy())
    pd.testing.assert_frame_equal(no_adder_df, adjusted_no_adder)


def test_formatted_db_adds_missing_columns():
    """Ensure formatted DataFrame contains expected columns"""

    df = pd.DataFrame(
        [
            {
                "feature": "Height",
                "summary": "Max height",
            }
        ]
    )
    formatted = finalize._formatted_db(df)
    assert list(formatted.columns) == finalize._PARSED_COLS
    assert len(formatted) == 1
    assert bool(formatted.iloc[0]["quantitative"]) is True


def test_extract_model_info_from_all_models_groups_tasks():
    """Group tasks by shared model configuration"""

    shared_model = DummyModelConfig(
        name="gpt",
        llm_call_kwargs={},
        llm_service_rate_limit=3,
        text_splitter_chunk_size=1500,
        text_splitter_chunk_overlap=100,
        client_type="openai",
    )
    other_model = DummyModelConfig(
        name="gpt-4",
        llm_call_kwargs={"temperature": 0.2},
        llm_service_rate_limit=1,
        text_splitter_chunk_size=1200,
        text_splitter_chunk_overlap=50,
        client_type="azure",
    )

    models = {
        "task_one": shared_model,
        "task_two": shared_model,
        "task_three": other_model,
    }

    info = finalize._extract_model_info_from_all_models(models)
    assert len(info) == 2

    first, second = info
    assert first["name"] == "gpt"
    assert first["tasks"] == ["task_one", "task_two"]
    assert first["llm_call_kwargs"] is None

    assert second["name"] == "gpt-4"
    assert second["tasks"] == ["task_three"]
    assert second["llm_call_kwargs"] == {"temperature": 0.2}


def test_compile_run_summary_message_includes_cost(tmp_path):
    """Include cost details when provided"""

    message = finalize.compile_run_summary_message(3661, 42.5, tmp_path, 3)
    assert "Total runtime: 1:01:01" in message
    assert "Total cost" in message
    assert "$42.50" in message
    assert "Number of jurisdictions with extracted data: 3" in message


def test_compile_run_summary_message_without_cost(tmp_path):
    """Omit cost line when not provided"""

    message = finalize.compile_run_summary_message(5, None, tmp_path, 0)
    assert "Total cost" not in message
    assert "Total runtime: 0:00:05" in message


def test_elapsed_time_as_str_basic():
    """Format elapsed time without days"""

    assert finalize._elapsed_time_as_str(65) == "0:01:05"


def test_elapsed_time_as_str_with_days():
    """Format elapsed time spanning days"""

    assert finalize._elapsed_time_as_str(90061) == "1 day, 1:01:01"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
