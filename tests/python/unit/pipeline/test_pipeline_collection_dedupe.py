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

    values = list(deduplicator.values())

    assert len(values) == 1
    assert values[0].from_steps == [
        "Look for document on jurisdiction website"
    ]
    assert deduplicator.info(doc).from_steps == [
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

    values = list(deduplicator.values())

    assert len(values) == 1
    assert values[0].doc is saved_doc
    assert values[0].from_steps == ["known_local_docs", "search_engine"]
    assert deduplicator.info(saved_doc).from_steps == [
        "known_local_docs",
        "search_engine",
    ]
    assert deduplicator.info(duplicate_doc).from_steps == [
        "known_local_docs",
        "search_engine",
    ]


@pytest.mark.parametrize("search_first", [False, True])
def test_search_duplicate_preserves_rank_and_source(search_first):
    """Search ranks merge without replacing the retained source"""
    deduplicator = DocumentDeDuplicator()
    known_doc = SimpleNamespace(
        attrs={
            "checksum": "shared",
            "source": "https://known.example",
            "collection_step_rank": 1,
            "doc_type": "html",
        }
    )
    search_doc = SimpleNamespace(
        attrs={
            "checksum": "shared",
            "source": "https://resolved.example",
            "doc_type": "html",
            "collection_step_rank": 3,
            "search_engines": ["test"],
        }
    )
    entries = [(known_doc, "known_doc_urls"), (search_doc, "search_engine")]
    if search_first:
        entries.reverse()

    for doc, step in entries:
        deduplicator.add_docs([doc], step_name=step)

    info = next(iter(deduplicator.values()))
    assert set(info.from_steps) == {"known_doc_urls", "search_engine"}
    assert info.doc.attrs["collection_step_rank"] == 3
    assert info.doc.attrs["source"] == (
        "https://resolved.example" if search_first else "https://known.example"
    )


def test_search_duplicates_keep_best_rank():
    """Equivalent search documents keep the highest ranked seed"""
    deduplicator = DocumentDeDuplicator()
    for rank in [4, 2, 3]:
        doc = SimpleNamespace(
            attrs={
                "checksum": "shared",
                "source": f"https://example.com/{rank}",
                "collection_step_rank": rank,
            }
        )
        deduplicator.add_docs([doc], step_name="search_engine")

    attrs = next(iter(deduplicator.values())).doc.attrs
    assert attrs["collection_step_rank"] == 2
    assert attrs["source"] == "https://example.com/4"


@pytest.mark.parametrize("incoming_rank", [2, 4, 6, None])
@pytest.mark.parametrize(
    "incoming_step", ["search_engine", "website_search_compass", None]
)
@pytest.mark.parametrize(
    "existing_engines, incoming_engines, expected_engines",
    [
        (
            ["google", "bing"],
            ["bing", "duckduckgo", "duckduckgo"],
            ["google", "bing", "duckduckgo"],
        ),
        (None, ["google"], ["google"]),
        (["google"], None, ["google"]),
        (None, None, None),
    ],
)
def test_search_duplicates_merge_engines_independently_of_rank(
    incoming_rank,
    incoming_step,
    existing_engines,
    incoming_engines,
    expected_engines,
):
    """Engine provenance accumulates regardless of step or rank"""
    deduplicator = DocumentDeDuplicator()
    original_engines = (
        list(existing_engines) if existing_engines is not None else None
    )
    retained_doc = SimpleNamespace(
        attrs={"checksum": "shared", "collection_step_rank": 4}
    )
    incoming_doc = SimpleNamespace(attrs={"checksum": "shared"})
    if existing_engines is not None:
        retained_doc.attrs["search_engines"] = existing_engines
    if incoming_engines is not None:
        incoming_doc.attrs["search_engines"] = incoming_engines
    if incoming_rank is not None:
        incoming_doc.attrs["collection_step_rank"] = incoming_rank

    deduplicator.add_docs([retained_doc], step_name="search_engine")
    deduplicator.add_docs([incoming_doc], step_name=incoming_step)

    info = next(iter(deduplicator.values()))
    assert len(deduplicator) == 1
    assert info.doc is retained_doc
    assert info.from_steps == (
        ["search_engine", incoming_step]
        if incoming_step not in {"search_engine", None}
        else ["search_engine"]
    )
    assert retained_doc.attrs["collection_step_rank"] == (
        min(4, incoming_rank)
        if incoming_step == "search_engine" and incoming_rank is not None
        else 4
    )
    assert retained_doc.attrs.get("search_engines") == expected_engines
    assert incoming_doc.attrs.get("search_engines") == incoming_engines
    assert existing_engines == original_engines


def test_non_search_duplicate_does_not_replace_search_rank():
    """Ranks from other collection steps cannot change seed eligibility"""
    deduplicator = DocumentDeDuplicator()
    search_doc = SimpleNamespace(
        attrs={
            "checksum": "shared",
            "collection_step_rank": 4,
        }
    )
    website_doc = SimpleNamespace(
        attrs={"checksum": "shared", "collection_step_rank": 1}
    )
    deduplicator.add_docs([search_doc], step_name="search_engine")
    deduplicator.add_docs([website_doc], step_name="website_search_compass")

    assert search_doc.attrs["collection_step_rank"] == 4


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
