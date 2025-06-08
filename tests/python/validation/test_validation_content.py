"""COMPASS Ordinance content validation tests"""

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

    class MockStructuredLLMCaller:
        """Mock LLM caller for tests."""

        async def call(self, key, text_chunk):
            """Mock LLM call and record system message"""
            keys.append(key)
            return text_chunk == 0

    text_chunks = list(range(7))
    validator = ParseChunksWithMemory(text_chunks, 3)
    caller = MockStructuredLLMCaller()

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


@flaky(max_runs=3, min_passes=1)
@pytest.mark.skipif(SHOULD_SKIP, reason="requires Azure OpenAI key")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_name,truth",
    [
        ("Hamilton New York.pdf", True),
        ("Decatur Indiana.pdf", True),
        ("ord_permit.pdf", False),
        ("model_ord.pdf", False),
        ("model_ord_pp.pdf", False),
        ("conservation_plan.pdf", False),
    ],
)
async def test_legal_text_validation(
    oai_llm_service, text_splitter, doc_loader, file_name, truth
):
    """Test using `LegalTextValidator` instance on documents"""

    legal_text_validator = LegalTextValidator(
        llm_service=oai_llm_service, temperature=0, seed=42, timeout=30
    )

    doc = doc_loader(file_name)
    chunks = text_splitter.split_text(doc.text)
    chunk_parser = ParseChunksWithMemory(chunks, num_to_recall=2)

    await parse_by_chunks(
        chunk_parser,
        heuristic=WindHeuristic(),
        legal_text_validator=legal_text_validator,
        callbacks=None,
        min_chunks_to_process=3,
    )

    assert legal_text_validator.is_legal_text == truth


@pytest.mark.skipif(
    SHOULD_SKIP or not PYT_CMD,
    reason="requires Azure OpenAI key *and* PyTesseract command to be set",
)
async def test_legal_text_validation_ocr(
    oai_llm_service, test_data_dir, text_splitter
):
    """Test the `LegalTextValidator` class for scanned doc"""
    import pytesseract  # noqa: PLC0415

    pytesseract.pytesseract.tesseract_cmd = PYT_CMD

    doc_fp = test_data_dir / "Sedgwick Kansas.pdf"
    with doc_fp.open("rb") as fh:
        pages = read_pdf_ocr(fh.read())
        doc = PDFDocument(pages)

    legal_text_validator = LegalTextValidator(
        llm_service=oai_llm_service, temperature=0, seed=42, timeout=30
    )

    chunks = text_splitter.split_text(doc.text)
    chunk_parser = ParseChunksWithMemory(chunks, num_to_recall=2)

    await parse_by_chunks(
        chunk_parser,
        heuristic=WindHeuristic(),
        legal_text_validator=legal_text_validator,
        callbacks=None,
        min_chunks_to_process=3,
    )

    assert legal_text_validator.is_legal_text


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
