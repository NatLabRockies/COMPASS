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


def test_add_docs_preserves_restored_artifacts_and_merges_provenance():
    """Restored docs should retain artifacts after duplicate discovery"""
    deduplicator = DocumentDeDuplicator()
    saved_doc = SimpleNamespace(
        attrs={
            "checksum": "abc123",
            "source": "https://example.com/ordinance.pdf",
            "source_fp": "source_docs/ordinance.pdf",
            "parsed_fp": "parsed_docs/ordinance.txt",
            "from_steps": ["known_local_docs"],
        }
    )
    duplicate_doc = SimpleNamespace(
        attrs={
            "checksum": "abc123",
            "source": "https://example.com/ordinance.pdf",
        }
    )

    deduplicator.add_docs([saved_doc])
    deduplicator.add_docs(
        [duplicate_doc],
        step_name="search_engine",
    )

    values = list(deduplicator.values)

    assert len(values) == 1
    assert values[0]["doc"] is saved_doc
    assert values[0]["from_steps"] == [
        "known_local_docs",
        "search_engine",
    ]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
