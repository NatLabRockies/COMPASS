"""Tests for collection document de-duplication"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from compass.pipeline.collection.dedupe import DocumentDeDuplicator


def test_add_docs_keeps_from_steps_unique_for_same_doc_and_step():
    """Repeated docs from one step should only record that step once"""
    deduplicator = DocumentDeDuplicator()
    doc = SimpleNamespace(attrs={"checksum": "abc123"})

    deduplicator.add_docs(
        [doc, doc],
        step_name="Look for document on jurisdiction website",
    )
    deduplicator.add_docs(
        [doc],
        step_name="Look for document on jurisdiction website",
    )

    values = list(deduplicator.values)

    assert len(values) == 1
    assert values[0]["from_steps"] == [
        "Look for document on jurisdiction website"
    ]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
