"""Checks for ranked traversal and evidence sampling."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from bs4 import BeautifulSoup
from elm.web.document import PDFDocument, HTMLDocument

from compass.web import priority_crawl as crawl
from compass.web.search import search_ordinance_candidates
from compass.utilities.url import canonical_url


def assessment(score=8, kind="ordinance", links=()):
    """Build an assessment fixture."""
    return {
        "title": "Data center rules",
        "document_score": score,
        "document_type": kind,
        "adoption_status": "adopted",
        "jurisdiction_evidence": "Example City",
        "technology_evidence": "Page 1: data centers",
        "adoption_evidence": "Page 1: adopted",
        "uncertainty": "",
        "links": list(links),
    }


def rating(link_id, score):
    """Build a link rating fixture."""
    return {"id": link_id, "score": score, "reason": "Official document link"}


@pytest.fixture
def loader():
    """Provide readers without external services."""
    return SimpleNamespace(
        content_fetcher=SimpleNamespace(get_kwargs={"ssl": False}),
        pdf_read_coroutine=AsyncMock(
            return_value=PDFDocument(
                [
                    "Example City enacted data center ordinance",
                ]
            )
        ),
        pdf_read_kwargs={},
        pdf_ocr_read_coroutine=None,
        html_loader=SimpleNamespace(
            fetch=AsyncMock(return_value=HTMLDocument([]))
        ),
    )


@pytest.fixture
def web(monkeypatch):
    """Capture fetch order through in-memory HTTP."""
    bodies, visited = {}, []

    def handle(request):
        url = str(request.url)
        visited.append(url)
        value = bodies[url]
        if isinstance(value, tuple):
            return httpx.Response(value[0], headers=value[1], request=request)
        return httpx.Response(200, content=value, request=request)

    client = httpx.AsyncClient
    monkeypatch.setattr(
        crawl.httpx,
        "AsyncClient",
        lambda **kw: client(
            **kw,
            transport=httpx.MockTransport(handle),
        ),
    )
    return bodies, visited


async def test_new_link_preempts_seeds_and_high_score_still_explores(
    web,
    loader,
    tmp_path,
):
    """A score above three must not prevent link expansion."""
    bodies, visited = web
    topic, pdf, other = [
        f"https://city.gov/{s}" for s in ["topic", "law.pdf", "other"]
    ]
    bodies[topic] = (
        f'<p>{"Data centers " * 30}</p><a href="law.pdf">Enacted ordinance</a>'
    )
    bodies[pdf] = b"%PDF-law"
    bodies[other] = (404, {})
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {"links": [rating(0, 8), rating(1, 6)]},
                assessment(5, "summary", [rating(0, 10)]),
                assessment(10),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, tmp_path
    )
    docs = await runner.run([{"url": topic}, {"url": other}])
    assert visited == [topic, pdf, other]
    assert docs[0].attrs["source"] == pdf
    assert docs[0].attrs["discovery"]["document_score"] == 10
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert trace["entries"][1]["state"] == "failed"
    assert "document_score" not in trace["entries"][1]
    assert trace["entries"][2]["parents"] == [topic]
    assert trace["stop_reason"] == "queue_exhausted"
    assert (tmp_path / "2" / "source.pdf").read_bytes() == bodies[pdf]


async def test_cycle_duplicate_bytes_and_invalid_link(web, loader, tmp_path):
    """Cycles and copies do not trigger repeated paid assessments."""
    bodies, visited = web
    a, b = "https://city.gov/a.pdf", "https://city.gov/b.pdf"
    bodies[a] = bodies[b] = b"%PDF-duplicate"
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {"links": [rating(0, 8), rating(1, 7)]},
                assessment(8, links=[rating(999, 10)]),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, tmp_path
    )
    docs = await runner.run([{"url": a}, {"url": b}])
    assert visited == [a, b]
    assert len(docs) == 1
    assert caller.call.await_count == 2
    assert runner.entries[a]["invalid_link_ids"] == [999]
    assert runner.entries[b]["duplicate_of"] == runner.entries[a]["id"]


async def test_page_limit_retains_queue(web, loader, tmp_path):
    """Reaching the fetch limit preserves unresolved candidates."""
    bodies, visited = web
    a, b = "https://city.gov/a.pdf", "https://city.gov/b.pdf"
    bodies[a] = b"%PDF-law"
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {"links": [rating(0, 8), rating(1, 7)]},
                assessment(7),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller,
        loader,
        {"technology": "data centers"},
        tmp_path,
        max_pages=1,
    )
    await runner.run([{"url": a}, {"url": b}])
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert visited == [a]
    assert trace["stop_reason"] == "page_limit"
    assert trace["remaining_queue"][0]["url"] == b


async def test_scanned_pdf_uses_ocr(web, loader, tmp_path):
    """Empty PDF text triggers the existing OCR reader."""
    url = "https://city.gov/law.pdf"
    web[0][url] = b"%PDF-scan"
    loader.pdf_read_coroutine.return_value = PDFDocument([])
    loader.pdf_ocr_read_coroutine = AsyncMock(
        return_value=PDFDocument(["Adopted data center rules"])
    )
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {"links": [rating(0, 10)]},
                assessment(10),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, tmp_path
    )
    assert len(await runner.run([{"url": url}])) == 1
    loader.pdf_ocr_read_coroutine.assert_awaited_once()


def test_long_pdf_and_unrelated_cover():
    """A technology clause deep in a code remains visible to the LLM."""
    pages = ["General zoning code"] * 536
    pages[112] = "Hyperscale data centers shall meet noise limits"
    pages[-1] = "Adopted by Example City Council"
    snippets = crawl._assessment_excerpts(PDFDocument(pages), "data centers")
    assert any(
        x["page"] == 113 and "noise limits" in x["text"] for x in snippets
    )
    assert snippets[-1]["page"] == 536


def test_links_preserve_pdf_urls_and_resolve_relative_paths():
    """Resolve observed document links without guessing URL spelling."""
    html = """<base href="https://city.gov/planning/">
      <a
        href="../archival-document?document=https%3A%2F%2Fs3.example%2FLaw%20A.pdf">
      Adopted ordinance</a>
      <a href="../council/calendar?id=42">Council record</a>
      <a href="javascript:void(0)">Menu</a>
      <iframe src="/viewer?documentId=123"></iframe>"""
    links = crawl._priority_page_links(
        BeautifulSoup(html, "html.parser"), "https://city.gov/topic"
    )
    assert [x["url"] for x in links] == [
        "https://s3.example/Law%20A.pdf",
        "https://city.gov/council/calendar?id=42",
        "https://city.gov/viewer?documentId=123",
    ]
    assert (
        canonical_url("https://CITY.gov/Law A.pdf?id=5&utm_source=g#page=3")
        == "https://city.gov/Law%20A.pdf?id=5"
    )
    assert canonical_url("https://city.gov/A+B.pdf") != canonical_url(
        "https://city.gov/A%20B.pdf"
    )
    assert canonical_url("https://city.gov/A.pdf") != canonical_url(
        "https://city.gov/a.pdf"
    )


async def test_search_merges_both_engines_and_preserves_snippets(monkeypatch):
    """Merge queries and engines without losing distinct documents."""
    call = AsyncMock(
        return_value=[
            [
                [
                    {
                        "url": "https://city.gov/law.pdf?utm_source=g",
                        "search_engine": "Google",
                        "attrs": {
                            "title": "Code",
                            "snippet": "Adopted data center rules",
                        },
                    }
                ],
                [{"url": "https://city.gov/zoning.pdf", "query": "zoning"}],
            ],
            [
                [
                    {
                        "url": "https://city.gov/law.pdf",
                        "search_engine": "DuckDuckGo",
                    }
                ]
            ],
        ]
    )
    monkeypatch.setattr("compass.web.search.search_all_se", call)
    queries = [
        "Example City data center ordinance", "Example City zoning ordinance",
    ]
    out = await search_ordinance_candidates(queries)
    assert len(out) == 2
    assert out[1]["url"] == "https://city.gov/zoning.pdf"
    assert call.call_args.args == (queries,)
    assert len(out[0]["sources"]) == 2
    assert (
        out[0]["sources"][0]["attrs"]["snippet"] == "Adopted data center rules"
    )
    assert call.call_args.kwargs["num_urls"] == 10
    assert call.call_args.kwargs["search_engines"] == [
        "SerpAPIGoogleSearch",
        "SerpAPIDuckDuckGoSearch",
    ]


async def test_depth_limit_and_cycle(web, loader, tmp_path):
    """The last permitted depth is assessed without expanding its links."""
    bodies, visited = web
    a, b = "https://city.gov/a", "https://city.gov/b"
    bodies[a] = f'<p>{"data center " * 30}</p><a href="/b">Adopted text</a>'
    bodies[b] = (
        f'<p>{"ordinance " * 30}</p><a href="/a">Source</a>'
        '<a href="/c">More</a>'
    )
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {"links": [rating(0, 10)]},
                assessment(2, "summary", [rating(0, 10)]),
                assessment(9, "ordinance", [rating(0, 10), rating(1, 9)]),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller,
        loader,
        {"technology": "data centers"},
        tmp_path,
        max_depth=1,
    )
    assert len(await runner.run([{"url": a}])) == 1
    assert visited == [a, b]
    assert len(runner.entries) == 2


def test_match_at_end_of_long_page_is_in_excerpt():
    """The excerpt window contains the technology match itself."""
    text = "General zoning. " * 1000 + "Data centers require a special permit."
    excerpts = crawl._assessment_excerpts(PDFDocument([text]), "data centers")
    assert "Data centers require a special permit" in excerpts[0]["text"]
    assert excerpts[0]["start_character"] > 0


async def test_unreadable_pdf_is_unassessed(web, loader, tmp_path):
    """No text is recorded as missing evidence rather than irrelevance."""
    url = "https://city.gov/law.pdf"
    web[0][url] = b"%PDF-empty"
    loader.pdf_read_coroutine.return_value = PDFDocument([])
    caller = SimpleNamespace(
        call=AsyncMock(return_value={"links": [rating(0, 10)]})
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, tmp_path
    )
    assert not await runner.run([{"url": url}])
    assert runner.entries[url]["state"] == "unreadable"
    assert runner.entries[url]["document_score"] is None
    assert caller.call.await_count == 1
