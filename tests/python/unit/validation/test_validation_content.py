"""COMPASS Ordinance content validation tests"""

import asyncio
import os
from pathlib import Path

import pytest
from flaky import flaky
from elm.web.document import PDFDocument
from elm.utilities.parse import read_pdf_ocr

from compass.extraction.wind.ordinance import WindHeuristic
from compass.validation.content import (
    parse_by_chunks,
    ParseChunksWithMemory,
    LegalTextValidator,
)


SHOULD_SKIP = os.getenv("AZURE_OPENAI_API_KEY") is None
PYT_CMD = os.getenv("TESSERACT_CMD")


@pytest.mark.asyncio
async def test_validation_with_mem():
    """Test the `ParseChunksWithMemory` class (basic execution)"""

    keys = []

    class MockJSONFromTextLLMCaller:
        """Mock LLM caller for tests"""

        async def call(self, key, text_chunk):
            """Mock LLM call and record system message"""
            keys.append(key)
            return text_chunk == 0

    text_chunks = list(range(7))
    validator = ParseChunksWithMemory(text_chunks, 3)
    caller = MockJSONFromTextLLMCaller()

    out = await validator.parse_from_ind(
        0, key="test", llm_call_callback=caller.call
    )
    assert out
    assert keys == ["test"]
    assert validator.memory == [{"test": True}, {}, {}, {}, {}, {}, {}]

    out = await validator.parse_from_ind(
        2, key="test", llm_call_callback=caller.call
    )
    assert out
    assert keys == ["test"] * 3
    assert validator.memory == [
        {"test": True},
        {"test": False},
        {"test": False},
        {},
        {},
        {},
        {},
    ]

    out = await validator.parse_from_ind(
        6, key="test", llm_call_callback=caller.call
    )
    assert not out
    assert keys == ["test"] * 6
    assert validator.memory == [
        {"test": True},
        {"test": False},
        {"test": False},
        {},
        {"test": False},
        {"test": False},
        {"test": False},
    ]


@pytest.mark.asyncio
async def test_parse_by_chunks_masks_chunk_after_successful_callback():
    """Test callback dispatch respects heuristic and callback results"""

    class MatchingHeuristic:
        """Recognize chunks explicitly marked as matching"""

        def check(self, text):
            """Return whether a chunk matches"""
            return text == "match"

    processed_indices = []

    async def callback(chunk_parser, ind):
        """Record processed chunk indices"""
        processed_indices.append(ind)
        await asyncio.sleep(0)
        return True

    chunk_parser = ParseChunksWithMemory(
        ["match", "skip", "match"], num_to_recall=2
    )

    await parse_by_chunks(
        chunk_parser,
        heuristic=MatchingHeuristic(),
        callbacks=[callback],
        min_chunks_to_process=0,
    )

    assert processed_indices == [0, 2]


@pytest.mark.asyncio
async def test_parse_by_chunks_stops_after_initial_invalid_chunks():
    """Test invalid initial chunks stop later callback processing"""

    class AlwaysMatchingHeuristic:
        """Recognize every chunk"""

        def check(self, text):
            """Return a matching result"""
            return True

    class InvalidTextValidator:
        """Reject every chunk and document"""

        def __init__(self):
            self.checked_indices = []

        async def check_chunk(self, chunk_parser, ind):
            """Record and reject a chunk"""
            self.checked_indices.append(ind)
            return False

        @property
        def is_correct_kind_of_text(self):
            """bool: Always reject the document"""
            return False

    processed_indices = []
    validator = InvalidTextValidator()

    async def callback(chunk_parser, ind):
        """Record processed chunk indices"""
        processed_indices.append(ind)
        await asyncio.sleep(0)
        return True

    chunk_parser = ParseChunksWithMemory(["one", "two", "three"])

    await parse_by_chunks(
        chunk_parser,
        heuristic=AlwaysMatchingHeuristic(),
        text_kind_validator=validator,
        callbacks=[callback],
        min_chunks_to_process=2,
    )

    assert validator.checked_indices == [0, 1]
    assert processed_indices == []


@flaky(max_runs=3, min_passes=1)
@pytest.mark.skipif(SHOULD_SKIP, reason="requires Azure OpenAI key")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_name,truth",
    [
        ("Johnson Iowa.pdf", True),
        ("Hamilton New York.pdf", True),
        ("Decatur Indiana.pdf", True),
        ("ord_permit.pdf", False),
        ("model_ord.pdf", False),
        ("model_ord_pp.pdf", False),
        ("conservation_plan.pdf", False),
        ("Rush_Indiana_draft.pdf", False),
    ],
)
async def test_legal_text_validation(
    oai_llm_service, text_splitter, doc_loader, file_name, truth
):
    """Test using `LegalTextValidator` instance on documents"""

    doc = doc_loader(file_name)
    legal_text_validator = LegalTextValidator(
        tech="wind",
        doc=doc,
        llm_service=oai_llm_service,
        temperature=0,
        seed=42,
        timeout=30,
    )

    chunks = text_splitter.split_text(doc.text)
    chunk_parser = ParseChunksWithMemory(chunks, num_to_recall=2)

    await parse_by_chunks(
        chunk_parser,
        heuristic=WindHeuristic(),
        text_kind_validator=legal_text_validator,
        callbacks=None,
        min_chunks_to_process=3,
    )

    assert legal_text_validator.is_correct_kind_of_text == truth


@flaky(max_runs=3, min_passes=1)
@pytest.mark.skipif(
    SHOULD_SKIP or not PYT_CMD,
    reason="requires Azure OpenAI key *and* PyTesseract command to be set",
)
async def test_legal_text_validation_ocr(
    oai_llm_service, test_data_files_dir, text_splitter
):
    """Test the `LegalTextValidator` class for scanned doc"""
    import pytesseract  # ruff:ignore[import-outside-top-level]

    pytesseract.pytesseract.tesseract_cmd = PYT_CMD

    doc_fp = test_data_files_dir / "Sedgwick Kansas.pdf"
    with doc_fp.open("rb") as fh:
        pages = read_pdf_ocr(fh.read())
        doc = PDFDocument(pages)

    doc.attrs["from_ocr"] = True

    legal_text_validator = LegalTextValidator(
        tech="wind",
        doc=doc,
        llm_service=oai_llm_service,
        temperature=0,
        seed=42,
        timeout=30,
    )

    chunks = text_splitter.split_text(doc.text)
    chunk_parser = ParseChunksWithMemory(chunks, num_to_recall=2)

    await parse_by_chunks(
        chunk_parser,
        heuristic=WindHeuristic(),
        text_kind_validator=legal_text_validator,
        callbacks=None,
        min_chunks_to_process=3,
    )

    assert legal_text_validator.is_correct_kind_of_text


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
