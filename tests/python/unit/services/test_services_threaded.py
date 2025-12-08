"""Test COMPASS Ordinances TempFileCache Services"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from elm.web.document import HTMLDocument

from compass.services import threaded
from compass.services.provider import RunningAsyncServices
from compass.services.threaded import (
    CleanedFileWriter,
    FileMover,
    HTMLFileLoader,
    JurisdictionUpdater,
    OrdDBFileWriter,
    TempFileCache,
    TempFileCachePB,
    ThreadedService,
    UsageUpdater,
    read_html_file,
)
from compass.utilities.logs import LocationFileLog, LogListener


logger = logging.getLogger("compass")


def _log_from_thread():
    """Call logger instance from a thread"""
    logger.debug("HELLO WORLD")


@pytest.mark.asyncio
async def test_temp_file_cache_service():
    """Test base implementation of `TempFileCache` class"""

    doc = HTMLDocument(["test"])
    doc.attrs["source"] = "http://www.example.com/?=%20test"

    cache = TempFileCache()
    cache.acquire_resources()
    out_fp = await cache.process(doc, doc.text)
    assert out_fp.exists()
    assert out_fp.read_text().startswith("test")
    cache.release_resources()
    assert not out_fp.exists()


@pytest.mark.asyncio
async def test_file_move_service(tmp_path):
    """Test base implementation of `FileMover` class"""

    doc = HTMLDocument(["test"])
    doc.attrs["source"] = "http://www.example.com/?=%20test"

    cache = TempFileCache()
    cache.acquire_resources()
    out_fp = await cache.process(doc, doc.text)
    assert out_fp.exists()
    assert out_fp.read_text().startswith("test")
    doc.attrs["cache_fn"] = out_fp

    date = datetime.now().strftime("%Y_%m_%d")
    expected_moved_fp = tmp_path / f"{out_fp.stem}_downloaded_{date}.txt"
    assert not expected_moved_fp.exists()
    mover = FileMover(tmp_path)
    mover.acquire_resources()
    moved_fp = await mover.process(doc)
    assert expected_moved_fp == moved_fp
    assert not out_fp.exists()
    assert moved_fp.exists()
    assert moved_fp.read_text().startswith("test")

    cache.release_resources()
    mover.release_resources()
    assert moved_fp.exists()


@pytest.mark.asyncio
async def test_logging_within_service(tmp_path, assert_message_was_logged):
    """Test that logging within a threaded service doesn't crash the process"""

    class ThreadedLogging(ThreadedService):
        """Subclass for testing"""

        @property
        def can_process(self):
            return True

        async def process(self):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.pool, _log_from_thread)

    log_listener = LogListener(["compass"], level="DEBUG")
    services = [ThreadedLogging()]

    async with RunningAsyncServices(services), log_listener as ll:
        with LocationFileLog(ll, tmp_path, location="test_loc", level="DEBUG"):
            await ThreadedLogging.call()

    assert_message_was_logged("HELLO WORLD")


def test_cache_file_with_hash_sets_source_and_checksum(tmp_path):
    """Ensure caching assigns a source when only source_fp is provided"""

    doc = HTMLDocument(["payload"])
    doc.attrs["source_fp"] = tmp_path / "source.pdf"

    cache_fp, checksum = threaded._cache_file_with_hash(
        doc, doc.text, tmp_path
    )

    assert cache_fp.exists()
    assert doc.attrs["source"] == str(doc.attrs["source_fp"])
    assert checksum == threaded._compute_sha256(cache_fp)
    assert cache_fp.read_text(encoding="utf-8") == doc.text


def test_cache_file_with_hash_generates_uuid_for_missing_source(tmp_path):
    """Source attr is generated when no source metadata exists"""

    doc = HTMLDocument(["payload"])
    cache_fp, checksum = threaded._cache_file_with_hash(
        doc, "text-data", tmp_path
    )

    assert cache_fp.exists()
    assert cache_fp.read_text() == "text-data"
    uuid_str = doc.attrs["source"]
    assert uuid_str

    uuid_obj = uuid.UUID(uuid_str)
    assert str(uuid_obj) == uuid_str
    assert checksum == threaded._compute_sha256(cache_fp)


def test_move_file_returns_none_without_cache(tmp_path):
    """_move_file should exit early when cache filename missing"""

    doc = HTMLDocument(["payload"])
    assert threaded._move_file(doc, tmp_path) is None


def test_move_file_uses_jurisdiction_name(tmp_path):
    """Verify moved filename uses jurisdiction specific naming"""

    cached_dir = tmp_path / "cached"
    cached_dir.mkdir()
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    cached_fp = cached_dir / "download.pdf"
    cached_fp.write_text("content", encoding="utf-8")

    doc = HTMLDocument(["payload"])
    doc.attrs.update(
        {"cache_fn": cached_fp, "jurisdiction_name": "Test County, ST"}
    )

    date = datetime.now().strftime("%Y_%m_%d")
    moved_fp = threaded._move_file(doc, out_dir)

    expected_name = f"Test_County_ST_downloaded_{date}.pdf"
    assert moved_fp.name == expected_name
    assert moved_fp.read_text(encoding="utf-8") == "content"
    assert not cached_fp.exists()


def test_move_file_handles_extensionless_cached_file(tmp_path):
    """Verify `_move_file` handles cached files without an extension"""

    cached_dir = tmp_path / "cached"
    cached_dir.mkdir()
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    cached_fp = cached_dir / "download"
    cached_fp.write_text("content", encoding="utf-8")

    doc = HTMLDocument(["payload"])
    doc.attrs["cache_fn"] = cached_fp

    date = datetime.now().strftime("%Y_%m_%d")
    moved_fp = threaded._move_file(doc, out_dir)

    assert moved_fp.name == f"{cached_fp.stem}_downloaded_{date}"
    assert moved_fp.read_text(encoding="utf-8") == "content"


def test_write_cleaned_file_with_debug(tmp_path, monkeypatch):
    """Cleaned file writer should emit cleaned and debug outputs"""

    doc = HTMLDocument(["payload"])
    doc.attrs.update(
        {
            "jurisdiction_name": "Sample Jurisdiction",
            "cleaned_ordinance_text": "clean",
            "districts_text": "districts",
            "ordinance_text": "orig",
            "permitted_use_text": "perm",
            "permitted_use_only_text": None,
        }
    )

    monkeypatch.setattr(threaded, "COMPASS_DEBUG_LEVEL", 1, raising=False)
    outputs = threaded._write_cleaned_file(doc, tmp_path)

    expected_files = {
        "Sample Jurisdiction Ordinance Summary.txt",
        "Sample Jurisdiction Districts.txt",
    }
    assert {fp.name for fp in outputs} == expected_files
    assert all(fp.exists() for fp in outputs)

    debug_fp = tmp_path / "Sample Jurisdiction Ordinance Original text.txt"
    assert debug_fp.exists()
    assert debug_fp.read_text(encoding="utf-8") == "orig"


def test_write_cleaned_file_without_jurisdiction_returns_none(tmp_path):
    """If jurisdiction name missing, cleaned file writer should do nothing"""

    doc = HTMLDocument(["payload"])
    doc.attrs["cleaned_ordinance_text"] = "clean"
    assert threaded._write_cleaned_file(doc, tmp_path) is None


def test_write_cleaned_file_skips_missing_section(tmp_path):
    """Missing sections should be skipped instead of erroring"""

    doc = HTMLDocument(["payload"])
    doc.attrs.update(
        {
            "jurisdiction_name": "Partial",
            "cleaned_ordinance_text": "clean",
        }
    )

    outputs = threaded._write_cleaned_file(doc, tmp_path)
    assert [fp.name for fp in outputs] == ["Partial Ordinance Summary.txt"]


def test_write_ord_db_creates_csv(tmp_path):
    """Ord database writer should create CSV output"""

    df = pd.DataFrame(
        {
            "feature": ["setback"],
            "value": ["10"],
            "summary": [""],
            "other": [1],
        }
    )
    doc = HTMLDocument(["payload"])
    doc.attrs.update(
        {
            "jurisdiction_name": "Sample",
            "scraped_values": df,
        }
    )

    out_fp = threaded._write_ord_db(doc, tmp_path)
    assert out_fp.exists()
    assert (
        out_fp.read_text(encoding="utf-8")
        == "feature,value,summary,other\nsetback,10,,1\n"
    )


def test_write_ord_db_requires_data(tmp_path):
    """Ord database writer returns None when data missing"""

    doc = HTMLDocument(["payload"])
    assert threaded._write_ord_db(doc, tmp_path) is None


@pytest.mark.asyncio
async def test_temp_file_cache_adds_source_and_checksum(tmp_path):
    """TempFileCache should populate source and checksum automatically"""

    doc = HTMLDocument(["payload"])
    doc.attrs["source_fp"] = tmp_path / "doc.html"

    cache = TempFileCache()
    cache.acquire_resources()
    out_fp = await cache.process(doc, "payload")

    assert doc.attrs["source"] == str(doc.attrs["source_fp"])
    assert doc.attrs["checksum"] == threaded._compute_sha256(out_fp)

    cache.release_resources()
    assert not out_fp.exists()


@pytest.mark.asyncio
async def test_temp_file_cache_pb_updates_progress(monkeypatch):
    """TempFileCachePB should advance progress bar using task name"""

    class DummyProgressBar:
        def __init__(self):
            self.calls = []

        def update_download_task(self, location, **kwargs):
            self.calls.append((location, kwargs))

    dummy_pb = DummyProgressBar()
    monkeypatch.setattr(threaded, "COMPASS_PB", dummy_pb, raising=False)

    doc = HTMLDocument(["payload"])
    doc.attrs["source"] = "http://example.com"

    async with RunningAsyncServices([TempFileCachePB()]):
        task = asyncio.current_task()
        original_name = task.get_name()
        try:
            task.set_name("Test Jurisdiction")
            out_fp = await TempFileCachePB.call(doc, doc.text)
        finally:
            task.set_name(original_name)

    assert dummy_pb.calls == [("Test Jurisdiction", {"advance": 1})]
    assert doc.attrs["checksum"].startswith("sha256:")
    assert not out_fp.exists()


@pytest.mark.asyncio
async def test_cleaned_file_writer_process(tmp_path, monkeypatch):
    """CleanedFileWriter should proxy through StoreFileOnDisk"""

    monkeypatch.setattr(threaded, "COMPASS_DEBUG_LEVEL", 0, raising=False)

    doc = HTMLDocument(["payload"])
    doc.attrs.update(
        {
            "jurisdiction_name": "Writer",
            "cleaned_ordinance_text": "clean",
            "districts_text": "district",
        }
    )

    writer = CleanedFileWriter(tmp_path)
    assert writer.can_process is True
    writer.acquire_resources()
    outputs = await writer.process(doc)
    writer.release_resources()

    assert sorted(fp.name for fp in outputs) == [
        "Writer Districts.txt",
        "Writer Ordinance Summary.txt",
    ]


@pytest.mark.asyncio
async def test_ord_db_file_writer_process(tmp_path):
    """OrdDBFileWriter should write csv using thread pool"""

    df = pd.DataFrame(
        {
            "feature": ["setback"],
            "value": ["10"],
            "summary": ["s"],
        }
    )
    doc = HTMLDocument(["payload"])
    doc.attrs.update(
        {
            "jurisdiction_name": "Ord",
            "scraped_values": df,
        }
    )

    writer = OrdDBFileWriter(tmp_path)
    writer.acquire_resources()
    out_fp = await writer.process(doc)
    writer.release_resources()

    assert out_fp.exists()


@pytest.mark.asyncio
async def test_usage_updater_process(tmp_path):
    """UsageUpdater should serialize tracker info to json"""

    class StubTracker:
        def __init__(self):
            self.add_called = False
            self.totals = {
                "gpt-4o": {
                    "prompt_tokens": 500_000,
                    "response_tokens": 250_000,
                }
            }

        def add_to(self, other):
            self.add_called = True
            other["stub"] = {"tracker_totals": self.totals}

    usage_fp = tmp_path / "usage.json"
    tracker = StubTracker()
    updater = UsageUpdater(usage_fp)
    updater.acquire_resources()

    assert updater.can_process is True
    usage_info = await updater.process(tracker)
    assert updater.can_process is True
    assert tracker.add_called is True

    updater.release_resources()

    with usage_fp.open(encoding="utf-8") as fh:
        persisted = json.load(fh)

    assert "stub" in persisted
    assert persisted == usage_info

    # Existing data path without tracker
    assert threaded._dump_usage(usage_fp, tracker=None) == persisted


@pytest.mark.asyncio
async def test_jurisdiction_updater_process(tmp_path):
    """JurisdictionUpdater should append jurisdiction entries"""

    jurisdiction_fp = tmp_path / "jurisdictions.json"
    updater = JurisdictionUpdater(jurisdiction_fp)
    updater.acquire_resources()

    assert updater.can_process is True
    updater._is_processing = True
    assert updater.can_process is False
    updater._is_processing = False

    jur1 = SimpleNamespace(
        full_name="Alpha County",
        county="Alpha",
        state="ST",
        subdivision_name="",
        type="county",
        code="00001",
    )

    await updater.process(jur1, None, 30, usage_tracker=None)

    with jurisdiction_fp.open(encoding="utf-8") as fh:
        data = json.load(fh)

    assert data["jurisdictions"][0]["found"] is False
    assert data["jurisdictions"][0]["documents"] is None
    assert data["jurisdictions"][0]["cost"] is None

    doc = HTMLDocument(["page"])
    doc.attrs.update(
        {
            "source": "http://example.com",
            "date": (2023, 6, 1),
            "out_fp": tmp_path / "ord" / "doc.pdf",
            "checksum": "sha256:abc",
            "from_ocr": True,
            "ordinance_text_ngram_score": 0.9,
            "permitted_use_text_ngram_score": 0.8,
            "jurisdiction_website": "http://jurisdiction.gov",
            "compass_crawl": True,
            "ordinance_values": pd.DataFrame(
                {
                    "feature": ["setback"],
                    "value": ["10"],
                    "summary": [""],
                }
            ),
        }
    )

    tracker = SimpleNamespace(
        totals={
            "gpt-4o": {
                "prompt_tokens": 1_000_000,
                "response_tokens": 2_000_000,
            }
        }
    )
    jur2 = SimpleNamespace(
        full_name="Beta City",
        county="Beta",
        state="ST",
        subdivision_name="Metro",
        type="city",
        code="00002",
    )

    await updater.process(jur2, doc, 12.5, tracker)

    with jurisdiction_fp.open(encoding="utf-8") as fh:
        data = json.load(fh)

    assert len(data["jurisdictions"]) == 2
    second = data["jurisdictions"][1]
    assert second["found"] is True
    assert second["jurisdiction_website"] == "http://jurisdiction.gov"
    assert second["compass_crawl"] is True
    assert pytest.approx(second["cost"]) == 22.5
    assert second["documents"][0]["ord_filename"] == "doc.pdf"
    assert second["documents"][0]["effective_year"] == 2023
    assert second["documents"][0]["num_pages"] == len(doc.pages)

    updater.release_resources()


def test_compute_jurisdiction_cost_uses_registry():
    """Ensure model costs are computed using registry values"""

    tracker = SimpleNamespace(
        totals={
            "gpt-4o": {
                "prompt_tokens": 1_000_000,
                "response_tokens": 1_000_000,
            }
        }
    )
    assert threaded._compute_jurisdiction_cost(tracker) == pytest.approx(12.5)

    tracker_unknown = SimpleNamespace(
        totals={"unknown": {"prompt_tokens": 1_000_000}}
    )
    assert threaded._compute_jurisdiction_cost(tracker_unknown) == 0


def test_dump_usage_without_tracker_returns_existing_data(tmp_path):
    """_dump_usage should return existing data unchanged when tracker absent"""

    usage_fp = tmp_path / "usage.json"
    initial = {"existing": True}
    usage_fp.write_text(json.dumps(initial), encoding="utf-8")

    assert threaded._dump_usage(usage_fp, tracker=None) == initial


@pytest.mark.asyncio
async def test_html_file_loader_and_read_html_file(tmp_path):
    """HTMLFileLoader should read files directly and via service queue"""

    html_fp = tmp_path / "doc.html"
    html_fp.write_text("<html><body>Hi</body></html>", encoding="utf-8")

    loader = HTMLFileLoader()
    loader.acquire_resources()
    doc, raw = await loader.process(html_fp)
    loader.release_resources()

    assert isinstance(doc, HTMLDocument)
    assert "Hi" in raw

    loader_service = HTMLFileLoader()
    async with RunningAsyncServices([loader_service]):
        task = asyncio.current_task()
        original_name = task.get_name()
        try:
            task.set_name("HTML Reader")
            doc_service, raw_service = await read_html_file(html_fp)
        finally:
            task.set_name(original_name)

    assert raw_service == raw
    assert doc_service.text == doc.text


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
