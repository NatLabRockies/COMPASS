"""Tests for the relevance check's decisions in event logs"""

import json

import pytest

from compass.utilities.event_log import EventLog, emit_event, use_event_log
from compass.validation.content import ParseChunksWithMemory, parse_by_chunks


class KeywordHeuristic:
    """Heuristic stand-in that passes chunks containing "keyword" """

    def check(self, text):
        return "keyword" in text


class LegalChunks:
    """Text-kind validator stand-in with fixed per-chunk verdicts"""

    def __init__(self, verdicts, document_is_legal=True):
        self.verdicts = verdicts
        self.document_is_legal = document_is_legal

    async def check_chunk(self, _chunk_parser, ind):
        emit_event("legal_check_ran")
        return self.verdicts[ind]

    @property
    def is_correct_kind_of_text(self):
        return self.document_is_legal


def collector(accepted):
    """Collector callback that accepts the chunks in `accepted`"""

    async def check_chunk(_chunk_parser, ind):  # ruff:ignore[unused-async]
        emit_event("collector_ran")
        return ind in accepted

    return check_chunk


def chunk_branches(events):
    """(chunk, step, outcome) of each branch event, in order"""
    return [
        (
            e["tags"]["chunk"],
            e["tags"]["step"].rpartition(".")[2],
            e["outcome"],
        )
        for e in events
        if e["type"] == "branch"
    ]


async def run_chunks(tmp_path, chunks, validator, accepted=()):
    """Scan `chunks` and return the events recorded"""
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        await parse_by_chunks(
            ParseChunksWithMemory(chunks, num_to_recall=2),
            KeywordHeuristic(),
            validator,
            callbacks=[collector(set(accepted))],
        )
    return [json.loads(x) for x in log.path.read_text().splitlines()]


@pytest.mark.asyncio
async def test_each_chunk_records_why_it_was_or_was_not_collected(tmp_path):
    """Head chunks get the legal check; later ones, the keyword check"""
    chunks = ["a", "b", "c", "d", "e", "f", "keyword g"]
    validator = LegalChunks([False, True, True, True, False])
    events = await run_chunks(tmp_path, chunks, validator, accepted={1})
    assert chunk_branches(events) == [
        (0, "head", "yes"),
        (0, "legal_tree", "not legal"),
        (1, "head", "yes"),
        (1, "legal_tree", "legal"),
        (1, "collect", "yes"),
        (2, "head", "yes"),
        (2, "legal_tree", "legal"),
        (2, "collect", "no"),
        (3, "head", "yes"),
        (3, "legal_tree", "legal"),
        (3, "collect", "no"),
        (4, "head", "yes"),
        (4, "legal_tree", "not legal"),
        # neither chunk 4 nor chunk 5 has a keyword, so 5 is skipped
        (5, "head", "no"),
        (5, "doc_legal", "yes"),
        (5, "heuristic", "no"),
        (6, "head", "no"),
        (6, "doc_legal", "yes"),
        (6, "heuristic", "yes"),
        (6, "collect", "no"),
    ]
    legal = [e for e in events if e["type"] == "legal_check_ran"]
    assert [e["tags"]["chunk"] for e in legal] == [0, 1, 2, 3, 4]
    assert {e["tags"]["step"] for e in legal} == {
        "filter.relevance.chunks.legal_tree"
    }
    collected = [e for e in events if e["type"] == "collector_ran"]
    assert [e["tags"]["chunk"] for e in collected] == [1, 2, 3, 6]
    assert {e["tags"]["chunks"] for e in collected} == {7}


@pytest.mark.asyncio
async def test_document_that_is_not_legal_text_stops_at_chunk_six(tmp_path):
    """Scanning stops once the first 5 chunks fail the legal check"""
    chunks = ["a"] * 8
    validator = LegalChunks([False] * 5, document_is_legal=False)
    events = await run_chunks(tmp_path, chunks, validator)
    assert chunk_branches(events)[-2:] == [
        (5, "head", "no"),
        (5, "doc_legal", "no: stop scanning"),
    ]
    assert max(e["tags"].get("chunk", 0) for e in events) == 5


@pytest.mark.asyncio
async def test_documents_exempt_from_the_legal_check(tmp_path):
    """Without a validator, head chunks go straight to the collectors"""
    events = await run_chunks(tmp_path, ["a", "b"], None, accepted={0})
    assert chunk_branches(events) == [
        (0, "head", "yes, legal check skipped"),
        (0, "collect", "yes"),
        (1, "head", "yes, legal check skipped"),
        (1, "collect", "no"),
    ]
