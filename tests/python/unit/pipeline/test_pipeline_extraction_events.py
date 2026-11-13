"""Tests for the decisions recorded around filtering, extraction, saving"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from elm.web.document import PDFDocument

import compass.pipeline.extraction.base as extraction_module
from compass.pipeline.extraction import DocumentExtraction
from compass.utilities.event_log import EventLog, use_event_log


class FakeExtractor:
    """Plugin stand-in that keeps (or drops) documents and finds data"""

    def __init__(self, keep, data):
        self.keep = keep
        self.data = data

    async def filter_docs(self, context, _max_docs):
        return context if self.keep else None

    async def parse_docs_for_structured_data(self, context):
        if self.data is None:
            return None
        context.attrs["structured_data"] = self.data
        return context


async def _write_db(_context, out_fn):  # ruff:ignore[unused-async]
    return f"/out/{out_fn}"


@pytest.fixture
def run_extraction(monkeypatch, tmp_path):
    """Run extract_from_docs and return the events it recorded"""
    monkeypatch.setattr(
        extraction_module,
        "COMPASS_PB",
        SimpleNamespace(update_jurisdiction_task=lambda *_a, **_k: None),
    )
    monkeypatch.setattr(
        extraction_module, "OrdDBFileWriter", SimpleNamespace(call=_write_db)
    )

    async def run(docs, keep=True, data=None):
        workflow = SimpleNamespace(
            jurisdiction=SimpleNamespace(full_name="Example County, Utah"),
            jurisdiction_website=None,
            extractor=FakeExtractor(keep, data),
            runtime=SimpleNamespace(
                request=SimpleNamespace(
                    parsing_settings=SimpleNamespace(
                        max_num_docs_per_jurisdiction=None
                    )
                )
            ),
        )
        log = EventLog(tmp_path / "events.jsonl")
        with use_event_log(log):
            out = await DocumentExtraction(workflow).extract_from_docs(docs)
        events = [json.loads(x) for x in log.path.read_text().splitlines()]
        return out, events

    return run


def _doc():
    doc = PDFDocument(["Data center ordinance text"])
    doc.attrs["source"] = "https://example.com/ord.pdf"
    return doc


def _branches(events):
    return [
        (e["tags"]["step"], e["outcome"])
        for e in events
        if e["type"] == "branch"
    ]


@pytest.mark.asyncio
async def test_step_without_documents_is_not_filtered(run_extraction):
    """No documents: nothing to filter, on to the next step"""
    out, events = await run_extraction([])
    assert out is None
    assert _branches(events) == [("handoff", "no")]


@pytest.mark.asyncio
async def test_nothing_kept_skips_extraction(run_extraction):
    """When filtering keeps nothing, extraction and saving never run"""
    out, events = await run_extraction([_doc()], keep=False)
    assert out is None
    assert _branches(events) == [
        ("handoff", "yes"),
        ("filter.any_kept", "no"),
    ]
    (kept,) = [e for e in events if e["type"] == "documents_kept"]
    assert kept["tags"]["step"] == "filter.order"
    assert not any(e.get("phase") == "extract" for e in events)


@pytest.mark.asyncio
async def test_extracted_data_is_saved(run_extraction):
    """Data found: the CSV is written and linked to its node"""
    data = pd.DataFrame([{"feature": "setback", "value": 100}])
    out, events = await run_extraction([_doc()], data=data)
    assert out is not None
    assert _branches(events) == [
        ("handoff", "yes"),
        ("filter.any_kept", "yes"),
        ("save.has_data", "yes"),
    ]
    (saved,) = [e for e in events if e["type"] == "saved"]
    assert saved["tags"]["step"] == "save.write"
    (rows,) = [e for e in events if e["type"] == "extracted_rows"]
    assert rows["tags"]["step"] == "extract"


@pytest.mark.asyncio
async def test_no_data_writes_nothing(run_extraction):
    """No data: the save step records that nothing was written"""
    _, events = await run_extraction([_doc()], data=None)
    assert _branches(events)[-1] == ("save.has_data", "no")
    (saved,) = [e for e in events if e["type"] == "saved"]
    assert saved["tags"]["step"] == "save.skip"
