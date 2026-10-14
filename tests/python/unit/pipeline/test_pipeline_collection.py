"""Tests for collection-step checkpoints"""

import asyncio
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace

import pytest

from compass.pipeline.collection.base import DocumentCollection
from compass.pipeline.jurisdiction import SingleJurisdictionRun
from compass.pipeline.collection.steps import (
    SearchResultsCrawlStep,
    _get_search_crawl_candidates,
)
from compass.pipeline.collection.persistence import (
    _write_collection_manifest_shard,
    _load_specific_collection_manifest_shard,
    load_specific_collection_manifest_shard,
    reset_collection_manifest_shard,
)
from compass.services.provider import RunningAsyncServices
from compass.services.threaded import GenericFuncRunner
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
        jurisdiction_website=None,
        known_local_docs=None,
        known_doc_urls=None,
        extraction=None,
        perform_se_search=True,
        num_search_results_to_crawl=0,
        runtime=SimpleNamespace(
            search_params=SimpleNamespace(num_search_results_to_crawl=0),
            re_run_targets=None,
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
            "results": documents,
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
    assert [doc["checksum"] for doc in workflow.writes[0]["results"]] == [
        "known"
    ]
    assert workflow.writes[0]["completed_step_document_counts"] == {
        "known_local_docs": 1
    }
    assert [doc["checksum"] for doc in workflow.writes[1]["results"]] == [
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
            "results": [_doc("known").attrs],
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
        "results": [_doc("known").attrs],
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
            "results": [],
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
        "results": [
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

    await workflow.collection._determine_unfinished_steps()
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
            "results": [],
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
    await collection._determine_unfinished_steps()

    assert list(collection._unfinished_steps()) == []


@pytest.mark.asyncio
async def test_collection_resume_keeps_persisted_docs_and_runs_new_step():
    """A partial checkpoint should retain old docs before the next step"""
    workflow = _workflow(
        {
            "results": [
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
    assert [doc["checksum"] for doc in collection_info["results"]] == [
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
            "results": [
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
    assert [doc["checksum"] for doc in collection_info["results"]] == ["known"]
    assert collection_info["completed_step_document_counts"] == {
        "known_local_docs": 1
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case, minimum, completed, old_count, expected_calls, reset",
    [
        (
            "passing",
            2,
            ["known_local_docs", "search_engine"],
            2,
            (0, 0),
            False,
        ),
        ("partial", 2, ["known_local_docs"], 1, (0, 1), False),
        ("partial_passing", 1, ["known_local_docs"], 1, (0, 1), False),
        ("failing", 2, ["known_local_docs", "search_engine"], 1, (1, 1), True),
        ("empty", 0, ["known_local_docs", "search_engine"], 0, (1, 1), True),
        ("missing", 2, [], 0, (1, 1), False),
    ],
)
async def test_targeted_collection_checkpoint_decisions(
    tmp_path, case, minimum, completed, old_count, expected_calls, reset
):
    """Resume partial shards and reset only complete failing shards"""
    shard_dir = tmp_path / "jurisdiction_dbs"
    jurisdiction = SimpleNamespace(
        code="12345", full_name="Example Township", website_url="configured"
    )
    existing = {
        "FIPS": jurisdiction.code,
        "full_name": jurisdiction.full_name,
        "num_docs": old_count,
        "num_search_results_to_crawl": 3,
        "jurisdiction_website": "saved",
        "results": [
            _doc(f"old-{index}", from_steps=["known_local_docs"]).attrs
            for index in range(old_count)
        ],
        "completed_step_document_counts": dict.fromkeys(completed, 0),
    }
    shard_fp = None
    if case != "missing":
        shard_fp = _write_collection_manifest_shard(shard_dir, existing)
    original_bytes = shard_fp.read_bytes() if shard_fp else None
    manifest_fp = tmp_path / "collection_manifest.json"
    manifest_fp.write_text("{}", encoding="utf-8")
    artifact_fp = tmp_path / "old.txt"
    artifact_fp.write_text("retained artifact", encoding="utf-8")
    workflow = _workflow()
    workflow.jurisdiction = jurisdiction
    workflow.jurisdiction_website = "configured"
    workflow.num_search_results_to_crawl = 10
    workflow.runtime.re_run_targets = {"num_docs": minimum}
    resets = []

    async def load_shard():
        info = await load_specific_collection_manifest_shard(
            shard_dir, jurisdiction
        )
        if info:
            workflow.jurisdiction_website = info["jurisdiction_website"]
        return info

    async def reset_shard():
        resets.append(True)
        await reset_collection_manifest_shard(
            shard_dir, tmp_path, jurisdiction
        )

    workflow.load_existing_collection_shard = load_shard
    workflow.reset_collection_shard = reset_shard
    known_docs = _Step("known_local_docs", [_doc("new-known")])
    search = _Step("search_engine", [_doc("new-search")])

    class _Collection(DocumentCollection):
        """Collection with deterministic steps that survive cache resets"""

        @cached_property
        def steps(self):
            """list: Deterministic collection steps"""
            return [known_docs, search]

    collection = _Collection(workflow)

    async with RunningAsyncServices([GenericFuncRunner()]):
        info = await collection.execute()

    assert (known_docs.calls, search.calls) == expected_calls
    assert bool(resets) is reset
    checksums = [doc["checksum"] for doc in info["results"]]
    if reset or case == "missing":
        assert checksums == ["new-known", "new-search"]
        assert workflow.num_search_results_to_crawl == 10
        assert workflow.jurisdiction_website == "configured"
    else:
        assert checksums[:old_count] == [
            f"old-{index}" for index in range(old_count)
        ]
        assert workflow.num_search_results_to_crawl == 3
        assert workflow.jurisdiction_website == "saved"
    if case == "passing":
        assert shard_fp.read_bytes() == original_bytes
        assert not workflow.writes
    assert manifest_fp.exists() is not reset
    assert artifact_fp.read_text(encoding="utf-8") == "retained artifact"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "saved_budget, configured_budget, crawl_completed, expected_budget, reset",
    [(0, 3, False, 3, True), (3, 0, True, 0, True), (3, 0, False, 3, False)],
)
async def test_collection_reset_refreshes_enabled_crawl_steps(
    tmp_path,
    saved_budget,
    configured_budget,
    crawl_completed,
    expected_budget,
    reset,
):
    """Full resets use current crawl settings while partial shards resume"""
    jurisdiction = SimpleNamespace(
        code="12345", full_name="Example Township", website_url="configured"
    )
    shard_dir = tmp_path / "shards"
    completed = {"known_local_docs": 1, "search_engine": 0}
    if crawl_completed:
        completed["search_results_crawl"] = 0
    _write_collection_manifest_shard(
        shard_dir,
        {
            "FIPS": jurisdiction.code,
            "full_name": jurisdiction.full_name,
            "num_docs": 1,
            "results": [_doc("old", from_steps=["known_local_docs"]).attrs],
            "completed_step_document_counts": completed,
            "num_search_results_to_crawl": saved_budget,
            "jurisdiction_website": "saved",
        },
    )
    runtime = SimpleNamespace(
        dirs=SimpleNamespace(jurisdiction_dbs=shard_dir, out=tmp_path),
        re_run_targets={"num_docs": 2},
    )
    workflow = SingleJurisdictionRun(
        runtime,
        jurisdiction,
        None,
        known_local_docs=[{"source_fp": "unused.txt"}],
        perform_se_search=True,
        perform_website_search=False,
        num_search_results_to_crawl=configured_budget,
    )

    async with RunningAsyncServices([GenericFuncRunner()]):
        await workflow.collection._determine_unfinished_steps()

    assert workflow.num_search_results_to_crawl == expected_budget
    step_names = [str(step.STEP_NAME) for step in workflow.collection.steps]
    assert ("search_results_crawl" in step_names) is (expected_budget > 0)
    assert bool(workflow.collection._collection_info) is not reset
    assert workflow.jurisdiction_website == (
        "configured" if reset else "saved"
    )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
