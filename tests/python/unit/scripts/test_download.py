"""Tests for compass.scripts.download"""

import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import AsyncExitStack

import pytest
from elm.web.document import HTMLDocument, MDDocument

import compass.scripts.download as download_module
from compass.scripts.download import (
    download_jurisdiction_ordinance_using_search_engine as search_docs,
    download_jurisdiction_ordinances_from_website_compass_crawl as crawl,
)
from compass.pipeline.data_classes import OutputSettings
from compass.pipeline.runtime import _setup_folders
from compass.services.provider import RunningAsyncServices
from compass.services.threaded import GenericFuncRunner
from compass.utilities.enums import LLMTasks


@pytest.mark.asyncio
@pytest.mark.parametrize("save_search_engine_results", [True, False])
@pytest.mark.parametrize("has_results", [True, False])
async def test_search_shards_written_to_disk(
    tmp_path, monkeypatch, save_search_engine_results, has_results
):
    """Persist complete search results only when shard saving is enabled"""
    dirs = _setup_folders(
        OutputSettings(
            tmp_path / "output",
            save_search_engine_results=save_search_engine_results,
        )
    )
    results = {
        "results": [
            {
                "url": "https://example.com/ordinance.pdf",
                "filtered_reason": "blocked_domain",
                "overall_rank": 1,
            }
        ]
        if has_results
        else []
    }

    async def fake_search(*_args, **_kwargs):  # ruff:ignore[unused-async]
        return results

    monkeypatch.setattr(
        download_module, "search_single_jurisdiction", fake_search
    )
    monkeypatch.setattr(
        download_module,
        "COMPASS_PB",
        SimpleNamespace(
            update_jurisdiction_task=lambda *_args, **_kwargs: None
        ),
    )
    async with RunningAsyncServices([GenericFuncRunner()]):
        docs = await search_docs(
            ["{jurisdiction} ordinance"],
            SimpleNamespace(full_name="Example County, Test"),
            se_shard_out_dir=dirs.se_shards,
        )

    assert docs == []
    shards = list(dirs.out.rglob("*_search_results.json"))
    if save_search_engine_results:
        assert len(shards) == 1
        assert shards[0].parent == dirs.se_shards
        assert json.loads(shards[0].read_text(encoding="utf-8")) == results
    else:
        assert shards == []
        assert not (dirs.out / "se_results").exists()


@pytest.mark.asyncio
async def test_find_jurisdiction_website_returns_base_domain(monkeypatch):
    """Return the canonical root URL for the selected website"""

    async def fake_search_with_fallback(**_kwargs):  # ruff:ignore[unused-async]
        return [
            "https://prattvilleal.gov/venue/autauga-county-commission/",
            "https://prattvilleal.gov/government/mayor",
            "https://example.org/other-page",
        ]

    class DummyValidator:
        def __init__(self, **_kwargs):
            pass

        async def check(self, url, jurisdiction):
            return url == "https://prattvilleal.gov/"

    monkeypatch.setattr(
        download_module,
        "search_with_fallback",
        fake_search_with_fallback,
    )
    monkeypatch.setattr(
        download_module,
        "JurisdictionWebsiteValidator",
        DummyValidator,
    )

    jurisdiction = SimpleNamespace(
        full_name="Autauga County, Alabama",
        full_name_the_prefixed="Autauga County, Alabama",
    )
    model_config = SimpleNamespace(
        llm_service=object(),
        llm_call_kwargs={},
    )

    out = await download_module.find_jurisdiction_website(
        jurisdiction, {LLMTasks.DEFAULT: model_config}
    )

    assert out == "https://prattvilleal.gov/"


@pytest.mark.asyncio
async def test_docs_from_web_search_adds_search_engine_attrs(monkeypatch):
    """Copy selected URL search engine provenance to document attrs"""

    async def fake_search_single_jurisdiction(  # ruff:ignore[unused-async]
        *_args, **_kwargs
    ):
        return {
            "results": [
                {
                    "url": "https://example.com/ordinance.pdf",
                    "overall_rank": 2,
                    "filtered_reason": None,
                    "search_engines": ["GoogleSearch", "BingSearch"],
                }
            ]
        }

    async def fake_docs_from_urls(  # ruff:ignore[unused-async]
        urls, *_args, **_kwargs
    ):
        assert urls == ["https://example.com/ordinance.pdf"]
        return [
            SimpleNamespace(
                attrs={"source": "https://example.com/ordinance.pdf"}
            )
        ]

    monkeypatch.setattr(
        download_module,
        "search_single_jurisdiction",
        fake_search_single_jurisdiction,
    )
    monkeypatch.setattr(
        download_module, "_docs_from_urls", fake_docs_from_urls
    )

    docs = await download_module._docs_from_web_search(
        query_templates=["{jurisdiction} ordinance"],
        num_urls=5,
        search_semaphore=None,
        browser_semaphore=None,
        url_ignore_substrings=None,
        jurisdiction=SimpleNamespace(full_name="Example County, Test"),
        simple_se_result_sort=False,
        se_shard_out_dir=None,
        search_results=None,
    )

    assert docs[0].attrs["collection_step_rank"] == 2
    assert docs[0].attrs["search_engines"] == ["GoogleSearch", "BingSearch"]


@pytest.mark.asyncio
async def test_search_candidate_budget_and_failed_downloads(monkeypatch):
    """Failed downloads preserve budgets and ranks without extra attrs"""
    urls = [f"https://example.com/{index}" for index in range(4)]
    requested = []

    async def search(*args, **kwargs):  # ruff:ignore[unused-async]
        assert args[2] == 3
        return {
            "results": [
                {
                    "url": url,
                    "overall_rank": index + 1,
                    "search_engines": ["test"],
                    "filtered_reason": (
                        None if index < args[2] else "beyond_top_n"
                    ),
                }
                for index, url in enumerate(urls)
            ]
        }

    async def fetch_doc(self, url):  # ruff:ignore[unused-async]
        requested.append(url)
        if url == urls[1]:
            raise OSError("failed candidate")
        doc_class = MDDocument if url == urls[0] else HTMLDocument
        doc = doc_class(pages=["Ordinance text"])
        doc.attrs.update(
            source=f"{url}/resolved",
            doc_type="pdf" if url == urls[0] else "html",
        )
        return doc, None

    monkeypatch.setattr(download_module, "search_single_jurisdiction", search)
    monkeypatch.setattr(
        download_module.COMPASSWebFileLoader, "_fetch_doc", fetch_doc
    )
    monkeypatch.setattr(
        download_module,
        "COMPASS_PB",
        SimpleNamespace(
            update_jurisdiction_task=lambda *_args, **_kwargs: None,
            file_download_prog_bar=lambda *_args: AsyncExitStack(),
        ),
    )
    docs = await download_module._docs_from_web_search(
        ["{jurisdiction}"],
        3,
        None,
        None,
        None,
        SimpleNamespace(full_name="Example"),
        True,
        None,
        None,
    )
    assert len(docs) == 2
    assert requested == urls[:3]
    assert [doc.attrs["doc_type"] for doc in docs] == ["pdf", "html"]
    assert all(
        not {"search_doc_type", "search_result_url", "search_resolved_url"}
        & doc.attrs.keys()
        for doc in docs
    )
    assert docs[1].attrs["source"] == urls[2]
    assert all("requested_url" not in doc.attrs for doc in docs)
    assert [doc.attrs["collection_step_rank"] for doc in docs] == [1, 3]
    assert docs[0].attrs["collection_step_rank"] == 1


@pytest.mark.asyncio
async def test_elm_crawl_tracks_accepted_partial_results(monkeypatch):
    """ELM crawl should retain accepted docs and completed pages early"""

    class DummyLoader:
        def __init__(self, **_kwargs):
            pass

    class DummyCrawler:
        def __init__(self, validator, **_kwargs):
            self.validator = validator

        async def run_with_timeout(
            self, _website, crawl_timeout_s, on_result_hook=None
        ):
            assert crawl_timeout_s == 3600
            result = SimpleNamespace(url="https://example.com/page")
            if on_result_hook:
                await on_result_hook(result)
            assert await self.validator(SimpleNamespace(text="keep", attrs={}))
            assert not await self.validator(
                SimpleNamespace(text="discard", attrs={})
            )
            return SimpleNamespace(
                documents=[SimpleNamespace(text="keep", attrs={})],
                raw_results=[result],
            )

    monkeypatch.setattr(download_module, "AsyncWebFileLoader", DummyLoader)
    monkeypatch.setattr(download_module, "COMPASSWebFileLoader", DummyLoader)
    monkeypatch.setattr(download_module, "ELMWebsiteCrawler", DummyCrawler)

    heuristic = SimpleNamespace(check=lambda text: text == "keep")

    (
        docs,
        results,
    ) = await download_module.download_jurisdiction_ordinances_from_website(
        "https://example.com",
        heuristic,
        {"ordinance": 1},
        return_c4ai_results=True,
    )

    assert [doc.text for doc in docs] == ["keep"]
    assert [result.url for result in results] == ["https://example.com/page"]


@pytest.mark.asyncio
async def test_compass_crawl_tracks_accepted_partial_results(monkeypatch):
    """COMPASS crawl should retain accepted docs before an early exit"""
    crawler_kwargs = {}

    class DummyCrawler:
        def __init__(self, validator, **kwargs):
            self.validator = validator
            crawler_kwargs.update(kwargs)

        async def run(self, _website, crawl_timeout_s, **_kwargs):
            assert crawl_timeout_s == 3600
            assert await self.validator(SimpleNamespace(text="keep", attrs={}))
            assert not await self.validator(
                SimpleNamespace(text="discard", attrs={})
            )
            return [SimpleNamespace(text="keep", attrs={})]

    monkeypatch.setattr(download_module, "COMPASSCrawler", DummyCrawler)

    heuristic = SimpleNamespace(check=lambda text: text == "keep")
    url_ignore_substrings = ["blocked.example"]
    url_keep_substrings = ["trusted.example"]

    docs = await crawl(
        "https://example.com",
        heuristic,
        {"ordinance": 1},
        url_ignore_substrings=url_ignore_substrings,
        url_keep_substrings=url_keep_substrings,
    )

    assert [doc.text for doc in docs] == ["keep"]
    assert crawler_kwargs["url_ignore_substrings"] is url_ignore_substrings
    assert crawler_kwargs["url_keep_substrings"] is url_keep_substrings


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
