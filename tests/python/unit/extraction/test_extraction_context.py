"""COMPASS extraction context tests"""

from pathlib import Path

import pytest
from elm.web.document import PDFDocument

from compass.extraction.context import ExtractionContext


def test_extraction_context_iter_empty():
    """Test empty ExtractionContext iteration"""
    for __ in ExtractionContext():
        msg = "Should not iterate over any documents"
        raise AssertionError(msg)


def test_extraction_context_iter_non_iterable():
    """Test non-iterable ExtractionContext iteration"""
    test_doc = PDFDocument([])
    for x in ExtractionContext(test_doc):
        assert isinstance(x, PDFDocument)
        assert x is test_doc


@pytest.mark.parametrize(
    "test_input", (("a", "b"), ["a", "b"], {"a": 1, "b": 2})
)
def test_extraction_context_iter_sequence(test_input):
    """Test non-sequence ExtractionContext iteration"""
    for x, y in zip(ExtractionContext(test_input), test_input, strict=True):
        assert x == y


def test_extraction_context_set():
    """Test non-sequence ExtractionContext iteration"""
    test_input = {"a", "b"}
    test = ExtractionContext(test_input)
    assert set(test) == test_input


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
