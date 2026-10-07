"""Tests for collection-step checkpoints"""

import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from compass.pipeline.collection.base import DocumentCollection
from compass.pipeline.collection.steps import (
    SearchResultsCrawlStep,
    _get_search_crawl_candidates,
)
from compass.pipeline.collection.persistence import (
    _write_collection_manifest_shard,
    _load_specific_collection_manifest_shard,
)
from compass.utilities.enums import COMPASSDocumentCollectionStep


class _Step:
    """Deterministic collection step for checkpoint tests"""

    def __init__(self, name, docs=None):
        self.STEP_NAME = name
        self.docs = docs or []
        self.calls = 0

    async def collect(self, workflow):
        """Record one collection attempt and return configured docs"""
        self.calls += 1
        return list(self.docs)


def _doc(checksum, **attrs):
    """Build a minimal collected document"""
    return SimpleNamespace(
        attrs={
            "checksum": checksum,
            "source": f"https://example.com/{checksum}.html",
            **attrs,
        }
    )


def _workflow(existing_collection_info=None):
    """Build a minimal collection workflow"""
    workflow = SimpleNamespace(
        jurisdiction=SimpleNamespace(full_name="Example Township"),
        extraction=None,
        runtime=SimpleNamespace(
            search_params=SimpleNamespace(num_search_results_to_crawl=0)
        ),
    )
    workflow.writes = []

    async def _load_existing_collection_shard():  # ruff:ignore[unused-async]
        return existing_collection_info

    async def _write_collection_shard_no_fail(deduplicator, completed_steps):
        await asyncio.sleep(0)
        documents = []
        for entry in deduplicator.values():
            document = dict(entry.doc.attrs)
            document["from_steps"] = list(entry.from_steps)
            documents.append(document)
        collection_info = {
            "documents": documents,
            "completed_step_document_counts": {
                step: sum(step in doc["from_steps"] for doc in documents)
                for step in completed_steps
            },
            "num_search_results_to_crawl": 3,
        }
        workflow.writes.append(deepcopy(collection_info))
        return collection_info

    workflow.load_existing_collection_shard = _load_existing_collection_shard
    workflow.write_collection_shard_no_fail = _write_collection_shard_no_fail
    return workflow


@pytest.mark.asyncio
async def test_collection_checkpoints_each_newly_completed_step():
    """Every newly completed step should write cumulative artifacts"""
    workflow = _workflow()
    known_docs = _Step("known_local_docs", [_doc("known")])
    search = _Step("search_engine", [_doc("search")])
    collection = DocumentCollection(workflow)
    collection.steps = [known_docs, search]

    collection_info = await collection.execute()

    assert known_docs.calls == 1
    assert search.calls == 1
    assert len(workflow.writes) == 2
    assert [doc["checksum"] for doc in workflow.writes[0]["documents"]] == [
        "known"
    ]
    assert workflow.writes[0]["completed_step_document_counts"] == {
        "known_local_docs": 1
    }
    assert [doc["checksum"] for doc in workflow.writes[1]["documents"]] == [
        "known",
        "search",
    ]
    assert workflow.writes[1]["completed_step_document_counts"] == {
        "known_local_docs": 1,
        "search_engine": 1,
    }
    assert collection_info == workflow.writes[-1]


@pytest.mark.asyncio
async def test_collection_skips_steps_recorded_in_existing_shard():
    """Persisted completed-step names should suppress repeat collection"""
    workflow = _workflow(
        {
            "documents": [_doc("known").attrs],
            "completed_step_document_counts": {
                "known_local_docs": 1,
                "search_engine": 0,
            },
        }
    )
    known_docs = _Step("known_local_docs", [_doc("known")])
    search = _Step("search_engine", [_doc("search")])
    collection = DocumentCollection(workflow)
    collection.steps = [known_docs, search]

    collection_info = await collection.execute()

    assert collection_info == {
        "documents": [_doc("known").attrs],
        "completed_step_document_counts": {
            "known_local_docs": 1,
            "search_engine": 0,
        },
    }
    assert known_docs.calls == 0
    assert search.calls == 0
    assert workflow.writes == []


@pytest.mark.asyncio
async def test_collection_restores_search_budgets_from_manifest():
    """First-run search budgets survive changes to runtime settings"""
    workflow = _workflow(
        {
            "documents": [],
            "completed_step_document_counts": {},
            "num_search_results_to_crawl": 3,
        }
    )
    collection = DocumentCollection(workflow)
    collection.steps = []
    workflow.runtime.search_params.num_search_results_to_crawl = 10

    await collection.execute()

    assert workflow.num_search_results_to_crawl == 3


@pytest.mark.asyncio
async def test_search_seeds_resume_from_disk_with_original_rank_cutoff(
    tmp_path,
):
    """Persisted search documents retain the original crawl eligibility"""
    jurisdiction = SimpleNamespace(code="12345", full_name="Example Township")
    info = {
        "FIPS": jurisdiction.code,
        "full_name": jurisdiction.full_name,
        "num_search_results_to_crawl": 4,
        "completed_step_document_counts": {"search_engine": 3},
        "documents": [
            _doc(
                str(rank),
                from_steps=["search_engine"],
                collection_step_rank=rank,
                doc_type=doc_type,
            ).attrs
            for rank, doc_type in [(1, "pdf"), (3, "html"), (5, "html")]
        ],
    }
    _write_collection_manifest_shard(tmp_path, info)
    saved = _load_specific_collection_manifest_shard(tmp_path, jurisdiction)
    workflow = _workflow(saved)
    workflow.runtime.search_params.num_search_results_to_crawl = 10
    workflow.collection = DocumentCollection(workflow)

    await workflow.collection._load_persisted_docs()
    seeds = _get_search_crawl_candidates(workflow)

    assert [seed["overall_rank"] for seed in seeds] == [3]
    assert seeds[0]["url"] == "https://example.com/3.html"
    assert workflow.num_search_results_to_crawl == 4


def test_search_results_crawl_step_order_is_independent_of_website_toggle():
    """Search result crawling follows search, even without website crawl"""
    workflow = _workflow()
    workflow.known_local_docs = []
    workflow.known_doc_urls = []
    workflow.perform_se_search = True
    workflow.perform_website_search = False
    workflow.perform_search_based_crawl = True
    workflow.runtime.search_params.num_search_results_to_crawl = 2

    collection = DocumentCollection(workflow)

    assert [step.STEP_NAME for step in collection.steps] == [
        COMPASSDocumentCollectionStep.SEARCH_ENGINE,
        COMPASSDocumentCollectionStep.SEARCH_RESULTS_CRAWL,
    ]
    assert isinstance(collection.steps[1], SearchResultsCrawlStep)


def test_search_results_crawl_step_is_not_registered_when_search_disabled():
    """Search result crawling is unavailable when search is disabled"""
    workflow = _workflow()
    workflow.known_local_docs = []
    workflow.known_doc_urls = []
    workflow.perform_se_search = False
    workflow.perform_website_search = False
    workflow.perform_search_based_crawl = False
    workflow.runtime.search_params.num_search_results_to_crawl = 2

    assert DocumentCollection(workflow).steps == []


async def test_completed_crawl_step_stays_completed_after_budget_change():
    """Changing N does not invalidate a completed collection checkpoint"""
    workflow = _workflow(
        {
            "documents": [],
            "completed_step_document_counts": {
                "search_results_crawl": 0,
                "search_engine": 0,
            },
            "num_search_results_to_crawl": 2,
        }
    )
    workflow.known_local_docs = []
    workflow.known_doc_urls = []
    workflow.perform_se_search = True
    workflow.perform_website_search = False
    workflow.perform_search_based_crawl = True
    workflow.runtime.search_params.num_search_results_to_crawl = 3
    collection = DocumentCollection(workflow)
    await collection._load_persisted_docs()

    assert list(collection._unfinished_steps()) == []


@pytest.mark.asyncio
async def test_collection_resume_keeps_persisted_docs_and_runs_new_step():
    """A partial checkpoint should retain old docs before the next step"""
    workflow = _workflow(
        {
            "documents": [
                _doc(
                    "known",
                    source_fp="source_docs/known.html",
                    parsed_fp="parsed_docs/known.txt",
                ).attrs
            ],
            "completed_step_document_counts": {"known_local_docs": 1},
        }
    )
    known_docs = _Step("known_local_docs", [_doc("known")])
    search = _Step("search_engine", [_doc("search")])
    collection = DocumentCollection(workflow)
    collection.steps = [known_docs, search]

    collection_info = await collection.execute()

    assert known_docs.calls == 0
    assert search.calls == 1
    assert [doc["checksum"] for doc in collection_info["documents"]] == [
        "known",
        "search",
    ]
    assert collection_info["completed_step_document_counts"] == {
        "known_local_docs": 0,
        "search_engine": 1,
    }


@pytest.mark.asyncio
async def test_collection_legacy_shard_runs_configured_steps():
    """Shards without completed-step counts should run configured steps"""
    workflow = _workflow(
        {
            "documents": [
                _doc(
                    "known",
                    source_fp="source_docs/known.html",
                    parsed_fp="parsed_docs/known.txt",
                ).attrs
            ]
        }
    )
    known_docs = _Step("known_local_docs", [_doc("known")])
    collection = DocumentCollection(workflow)
    collection.steps = [known_docs]

    collection_info = await collection.execute()

    assert known_docs.calls == 1
    assert [doc["checksum"] for doc in collection_info["documents"]] == [
        "known"
    ]
    assert collection_info["completed_step_document_counts"] == {
        "known_local_docs": 1
    }


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
