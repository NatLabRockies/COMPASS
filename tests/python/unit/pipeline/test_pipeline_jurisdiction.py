"""Tests for compass.pipeline.jurisdiction"""

import asyncio
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

import compass.pipeline.jurisdiction as jurisdiction_module
from compass.exceptions import COMPASSPluginConfigurationError
from compass.pipeline.jurisdiction import SingleJurisdictionRun


class _NoOpLocationFileLog:
    """No-op location log context for wrapper tests"""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def jurisdiction_run(monkeypatch, tmp_path):
    """Create a minimal jurisdiction run for wrapper testing"""

    monkeypatch.setattr(
        jurisdiction_module, "DocumentExtraction", lambda _: None
    )
    monkeypatch.setattr(
        jurisdiction_module, "DocumentCollection", lambda _: None
    )
    monkeypatch.setattr(
        jurisdiction_module, "LocationFileLog", _NoOpLocationFileLog
    )
    monkeypatch.setattr(
        jurisdiction_module.COMPASS_PB,
        "jurisdiction_prog_bar",
        lambda _: nullcontext(),
    )

    runtime = SimpleNamespace(
        jurisdiction_semaphore=asyncio.Semaphore(1),
        log_listener=None,
        dirs=SimpleNamespace(logs=tmp_path),
        log_level="INFO",
    )
    jurisdiction = SimpleNamespace(
        full_name="Test County, Colorado",
        code="08001",
        website_url="https://example.com",
    )
    extractor = SimpleNamespace()
    return SingleJurisdictionRun(runtime, jurisdiction, extractor)


@pytest.mark.asyncio
async def test_run_process_with_logging_reraises_plugin_config_errors(
    monkeypatch, jurisdiction_run
):
    """Plugin configuration errors should propagate out of logging wrapper"""

    async def _raise_config_error(self):  # ruff:ignore[unused-async]
        raise COMPASSPluginConfigurationError("bad plugin config")

    monkeypatch.setattr(SingleJurisdictionRun, "process", _raise_config_error)

    with pytest.raises(
        COMPASSPluginConfigurationError, match="bad plugin config"
    ):
        await jurisdiction_run.run_process_with_logging()


@pytest.mark.asyncio
async def test_collect_executes_collection_workflow(jurisdiction_run):
    """Collection should delegate to the collection workflow"""
    collection_info = {"FIPS": "08001", "documents": []}
    captured = {}

    class _CollectionWorkflow:
        """Capture collection execution inputs"""

        async def execute(self, **kwargs):
            captured.update(kwargs)
            return collection_info

    jurisdiction_run.collection = _CollectionWorkflow()

    out = await jurisdiction_run.collect()

    assert out == collection_info
    assert captured == {"eager_extract": False}


@pytest.mark.asyncio
async def test_load_existing_collection_shard_restores_website(
    monkeypatch, jurisdiction_run, tmp_path
):
    """A loaded shard should restore its jurisdiction website"""
    collection_info = {
        "FIPS": "08001",
        "jurisdiction_website": "https://saved.example.com",
        "completed_step_document_counts": {"known_local_docs": 0},
        "documents": [],
    }

    async def _load_shard(*args):  # ruff:ignore[unused-async]
        return collection_info

    jurisdiction_run.runtime.dirs.jurisdiction_dbs = tmp_path / "shards"
    monkeypatch.setattr(
        jurisdiction_module,
        "load_specific_collection_manifest_shard",
        _load_shard,
    )

    out = await jurisdiction_run.load_existing_collection_shard()

    assert out == collection_info
    assert jurisdiction_run.jurisdiction_website == "https://saved.example.com"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
