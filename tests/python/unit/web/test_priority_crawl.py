"""Checks for ranked traversal and evidence sampling."""

import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from bs4 import BeautifulSoup
from elm.web.document import PDFDocument, HTMLDocument

from compass.llm.calling import SchemaOutputLLMCaller
from compass.web import priority_crawl as crawl
from compass.web.search import search_ordinance_candidates
from compass.utilities.event_log import EventLog
from compass.utilities.url import canonical_url


def assessment(
    score=8, kind="ordinance", links=(), date=None, status="adopted"
):
    """Build an assessment fixture."""
    return {
        "title": "Data center rules",
        "document_score": score,
        "document_type": kind,
        "adoption_status": status,
        "jurisdiction_evidence": "Example City",
        "technology_evidence": "Page 1: data centers",
        "adoption_evidence": "Page 1: adopted",
        "latest_date": date
        or {
            "year": None,
            "month": None,
            "day": None,
            "evidence": "",
        },
        "uncertainty": "",
        "links": list(links),
    }


def rating(link_id, score):
    """Build a link rating fixture."""
    return {"id": link_id, "score": score, "reason": "Official document link"}


class _Service:
    """LLM service that returns the given answers in order, as JSON"""

    model_name = "test-model"

    def __init__(self, *answers):
        self.answers = list(answers)

    async def call(self, **_):
        return json.dumps(self.answers.pop(0))


class _FakeFetcher:
    """Stand-in for ELM's content fetcher, serving pages from memory

    Pages are described with ``httpx`` responses, so redirects and error
    statuses behave as they would over the network. Each response is
    reported to ``on_response_hook`` the way ELM's fetcher does.
    """

    def __init__(self):
        self.bodies, self.visited = {}, []
        self.on_response_hook = self.on_error_hook = None

    def _respond(self, request):
        url = str(request.url)
        self.visited.append(url)
        value = self.bodies[url]
        if callable(value):
            return value(request)
        if isinstance(value, tuple):
            return httpx.Response(value[0], headers=value[1], request=request)
        return httpx.Response(200, content=value, request=request)

    async def fetch(self, url):
        transport = httpx.MockTransport(self._respond)
        async with httpx.AsyncClient(
            transport=transport, follow_redirects=True
        ) as client:
            try:
                response = await client.get(url)
            except httpx.RequestError as error:
                # as ELM does once its retries run out
                await self.on_error_hook(url, error)
                return None
        await self.on_response_hook(
            url, _aiohttp_response(response), response.content
        )
        content_type = response.headers.get("content-type", "")
        return response.content, content_type, "utf-8", response.headers


def _aiohttp_response(response):
    """The parts of an aiohttp response that are read from the hook"""
    return SimpleNamespace(
        status=response.status_code,
        reason=response.reason_phrase,
        url=response.url,
        headers=response.headers,
        history=[
            SimpleNamespace(status=h.status_code, url=h.url, headers=h.headers)
            for h in response.history
        ],
        request_info=SimpleNamespace(
            method=response.request.method,
            url=response.request.url,
            headers=response.request.headers,
        ),
    )


@pytest.fixture
def loader(monkeypatch):
    """Provide readers without external services.

    ``render`` stands in for the browser: it returns the HTML a page
    renders to, empty by default.
    """
    render = AsyncMock(return_value="")
    monkeypatch.setattr(crawl, "load_html_with_pw", render)
    return SimpleNamespace(
        render=render,
        content_fetcher=_FakeFetcher(),
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
            browser_semaphore=None,
            PAGE_LOAD_TIMEOUT=1_000,
            uss=False,
            pw_launch_kwargs={},
        ),
    )


@pytest.fixture
def web(loader):
    """Pages served to the crawler, and the order they were fetched in"""
    fetcher = loader.content_fetcher
    return fetcher.bodies, fetcher.visited


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
        caller,
        loader,
        {"technology": "data centers"},
        events=_log(tmp_path),
        stop_score=None,
    )
    docs = await runner.run([{"url": topic}, {"url": other}])
    assert visited == [topic, pdf, other]
    assert docs[0].attrs["source"] == pdf
    assert docs[0].attrs["discovery"]["document_score"] == 10
    assert runner.entries[other]["state"] == "failed"
    assert "document_score" not in runner.entries[other]
    assert runner.entries[pdf]["parents"] == [topic]
    assert _events(tmp_path)[-1]["stop_reason"] == "queue_exhausted"
    assert docs[0].attrs["raw_pdf"] == bodies[pdf]
    assert list(tmp_path.iterdir()) == [tmp_path / "events.jsonl"]


async def test_crawl_runs_without_an_event_log(web, loader):
    """The crawler works, and records nothing, when given no event log"""
    bodies, _ = web
    page = "https://city.gov/law"
    bodies[page] = f"<p>{'Data centers ' * 30}</p>"
    caller = SimpleNamespace(
        call=AsyncMock(side_effect=[{"links": [rating(0, 9)]}, assessment(10)])
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}
    )
    docs = await runner.run([{"url": page}])
    assert [doc.attrs["source"] for doc in docs] == [page]


async def test_unanswered_request_is_retried_in_a_browser(
    web, loader, tmp_path, monkeypatch
):
    """A page that never answers is given up on and tried in a browser"""
    page = "https://city.gov/slow"

    async def never_answers(_url):
        await asyncio.sleep(60)

    monkeypatch.setattr(crawl, "_FETCH_TIMEOUT", 0.01)
    monkeypatch.setattr(loader.content_fetcher, "fetch", never_answers)
    loader.render.return_value = f"<p>{'Data centers ' * 30}</p>"
    caller = SimpleNamespace(
        call=AsyncMock(side_effect=[{"links": [rating(0, 9)]}, assessment(10)])
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    docs = await runner.run([{"url": page}])

    assert [doc.attrs["source"] for doc in docs] == [page]
    http, browser = [e for e in _events(tmp_path) if e["type"] == "fetched"]
    assert "no response within" in http["error"]
    assert browser["via"] == "browser"


async def test_failed_connection_logs_its_reason(web, loader, tmp_path):
    """The error behind a request with no response reaches the log"""
    bodies, _ = web
    page = "https://city.gov/down.pdf"

    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    bodies[page] = refuse
    caller = SimpleNamespace(
        call=AsyncMock(return_value={"links": [rating(0, 9)]})
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    assert await runner.run([{"url": page}]) == []

    (http,) = [e for e in _events(tmp_path) if e["type"] == "fetched"]
    assert http["error"] == "ConnectError: connection refused"
    (failed,) = [e for e in _events(tmp_path) if e["type"] == "visit_failed"]
    assert "ConnectError: connection refused" in failed["error"]


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
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    docs = await runner.run([{"url": a}, {"url": b}])
    assert visited == [a, b]
    assert len(docs) == 1
    assert caller.call.await_count == 2
    assert runner.entries[a]["link_problems"] == [
        {"id": 999, "issue": "unknown link id"}
    ]
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
        events=_log(tmp_path),
        max_pages=1,
    )
    await runner.run([{"url": a}, {"url": b}])
    finished = _events(tmp_path)[-1]
    assert visited == [a]
    assert finished["stop_reason"] == "page_limit"
    assert finished["remaining_queue"][0]["url"] == b


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
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
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
        "Example City data center ordinance",
        "Example City zoning ordinance",
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
        events=_log(tmp_path),
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
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    assert not await runner.run([{"url": url}])
    assert runner.entries[url]["state"] == "unreadable"
    assert runner.entries[url]["document_score"] is None
    assert caller.call.await_count == 1


async def test_ranking_tolerates_omitted_and_invented_links(
    web, loader, tmp_path
):
    """Bad ranking output is recorded instead of aborting the search."""
    bodies, visited = web
    a, b = "https://city.gov/a.pdf", "https://city.gov/b.pdf"
    bodies[a] = b"%PDF-a"
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {"links": [rating(0, 8), rating(7, 9)]},
                assessment(8),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    docs = await runner.run([{"url": a}, {"url": b}])
    assert visited == [a]
    assert len(docs) == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    resolved = next(e for e in events if e["type"] == "ranking_resolved")
    assert resolved["selected"] == [a]
    assert resolved["not_selected"] == [b]
    assert resolved["problems"] == [{"id": 7, "issue": "unknown link id"}]


async def test_failed_ranking_falls_back_to_search_order(
    web, loader, tmp_path
):
    """A truncated or empty ranking still crawls, in search-rank order."""
    bodies, visited = web
    a, b = "https://city.gov/a.pdf", "https://city.gov/b.pdf"
    bodies[a], bodies[b] = b"%PDF-a", b"%PDF-b"
    caller = SimpleNamespace(
        call=AsyncMock(side_effect=[{}, assessment(8), assessment(9)])
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    seeds = [
        {"url": a, "sources": [{"query": "q", "query_rank": 4}]},
        {"url": b, "sources": [{"query": "q", "query_rank": 1}]},
    ]
    docs = await runner.run(seeds)
    assert visited == [b, a]
    assert len(docs) == 2
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    fallback = next(e for e in events if e["type"] == "ranking_fallback")
    assert fallback["order"] == [b, a]


async def test_events_replay_the_full_search(web, loader, tmp_path):
    """The event log records each step in order with full payloads."""
    bodies, _ = web
    a = "https://city.gov/a.pdf"
    bodies[a] = b"%PDF-a"
    caller = SchemaOutputLLMCaller(
        _Service({"links": [rating(0, 9)]}, assessment(8))
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    await runner.run([{"url": a, "sources": [{"query": "q"}]}])
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    # Each decision of the crawl is recorded against the workflow map
    assert [
        (e["tags"]["step"].rsplit(".", 1)[1], e["outcome"])
        for e in events
        if e["type"] == "branch"
    ] == [
        ("loaded", "yes"),
        ("new", "yes"),
        ("assess", "assessed"),
        ("candidate", "yes"),
        ("follow", "yes"),
        ("stop", "stop"),
    ]
    # schemas are logged once each, ahead of the first call that uses them
    events = [e for e in events if e["type"] not in {"branch", "llm_schema"}]
    assert [(e["phase"], e["type"]) for e in events] == [
        (None, "run_started"),
        ("search", "phase_started"),
        ("search", "search_results"),
        ("search", "llm_request"),
        ("search", "llm_response"),
        ("search", "ranking_resolved"),
        ("search", "enqueued"),
        ("search", "phase_finished"),
        ("crawl", "phase_started"),
        ("crawl", "visit_started"),
        ("crawl", "fetched"),
        ("crawl", "page_read"),
        ("crawl", "llm_request"),
        ("crawl", "llm_response"),
        ("crawl", "assessed"),
        ("crawl", "phase_finished"),
        (None, "run_finished"),
    ]
    events = [e for e in events if not e["type"].startswith("phase_")]
    ranking_request, assess_request = events[2], events[9]
    assert ranking_request["tags"]["step"].endswith("priority.rank")
    sent = json.loads(ranking_request["messages"][1]["content"])
    assert sent["links"][0]["url"] == a
    assert json.loads(events[3]["response"]) == {"links": [rating(0, 9)]}
    # an assessment is tied to the queue entry it judged
    assert assess_request["tags"]["step"].endswith("priority.assess")
    assert assess_request["tags"]["entry_id"] == 0
    assert events[6]["queue"][0]["url"] == a
    assert events[11]["accepted_as_candidate"] is True
    assert events[-1]["candidates"][0]["url"] == a


async def test_empty_ranking_falls_back_to_search_order(
    web, loader, tmp_path, caplog
):
    """An unusable ranking answer is logged and the search order used."""
    bodies, visited = web
    a = "https://city.gov/a.pdf"
    bodies[a] = b"%PDF-a"
    caller = SimpleNamespace(call=AsyncMock(side_effect=[{}, assessment(8)]))
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    await runner.run([{"url": a}])

    assert any(e["type"] == "ranking_fallback" for e in _events(tmp_path))
    assert visited == [a]
    assert "no usable answer" in caplog.text


async def test_only_top_ranked_links_are_queued(web, loader, tmp_path):
    """Links beyond max_ranked_links are not queued, best scores first."""
    bodies, visited = web
    urls = [f"https://city.gov/{i}.pdf" for i in range(4)]
    for url in urls:
        bodies[url] = b"%PDF-" + url.encode()
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {
                    "links": [
                        rating(0, 3),
                        rating(1, 9),
                        rating(2, 7),
                        rating(3, 5),
                    ]
                },
                assessment(0, "other"),
                assessment(0, "other"),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller,
        loader,
        {"technology": "data centers"},
        events=_log(tmp_path),
        max_ranked_links=2,
    )
    await runner.run([{"url": url} for url in urls])
    assert set(runner.entries) == {urls[1], urls[2]}
    assert visited == [urls[1], urls[2]]
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    queued = [e for e in events if e["type"] == "enqueued"]
    assert len(queued) == 1
    assert queued[0]["source"] == "ranking"
    assert [x["url"] for x in queued[0]["items"]] == [urls[1], urls[2]]


async def test_search_keeps_top_results_per_engine(monkeypatch):
    """Each engine contributes its top unique, unblocked URLs by rank."""

    def hits(engine, query, urls):
        return [
            {
                "url": url,
                "search_engine": engine,
                "query": query,
                "query_rank": i,
            }
            for i, url in enumerate(urls, 1)
        ]

    call = AsyncMock(
        return_value=[
            [
                hits(
                    "G",
                    "q1",
                    [
                        "https://a.gov/1",
                        "https://facebook.com/x",
                        "https://a.gov/2",
                    ],
                ),
                hits(
                    "G",
                    "q2",
                    ["https://b.gov/1", "https://a.gov/1", "https://b.gov/2"],
                ),
            ],
            [hits("D", "q1", ["https://c.gov/1", "https://a.gov/1"])],
        ]
    )
    monkeypatch.setattr("compass.web.search.search_all_se", call)
    out = await search_ordinance_candidates(
        ["q1", "q2"],
        results_per_engine=3,
        url_ignore_substrings=["facebook.com"],
    )
    urls = [c["url"] for c in out]
    assert urls == [
        "https://a.gov/1",
        "https://b.gov/1",
        "https://a.gov/2",
        "https://c.gov/1",
    ]
    by_url = {c["url"]: c for c in out}
    assert [
        s["search_engine"] for s in by_url["https://a.gov/1"]["sources"]
    ] == [
        "G",
        "D",
    ]


async def test_crawl_stops_once_a_document_reaches_stop_score(
    web, loader, tmp_path
):
    """A score-10 ordinance ends the crawl before lower-priority pages."""
    bodies, visited = web
    a, b = "https://city.gov/a.pdf", "https://city.gov/b.pdf"
    bodies[a], bodies[b] = b"%PDF-a", b"%PDF-b"
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {"links": [rating(0, 9), rating(1, 8)]},
                assessment(10),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    docs = await runner.run([{"url": a}, {"url": b}])
    assert visited == [a]
    assert len(docs) == 1
    finished = json.loads(
        (tmp_path / "events.jsonl").read_text().splitlines()[-1]
    )
    assert finished["stop_reason"] == "found_document"
    assert [x["url"] for x in finished["remaining_queue"]] == [b]


async def test_blocked_page_is_retried_in_a_browser(web, loader, tmp_path):
    """A 403 page is rendered in the browser and still assessed."""
    bodies, visited = web
    page = "https://code.example/enactments/Ord-1"
    bodies[page] = (403, {})
    rendered = (
        f"<p>{'Data center ordinance text. ' * 20}</p>"
        '<a href="/ord-1.pdf">Ordinance PDF</a>'
    )
    loader.render.return_value = rendered
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[{"links": [rating(0, 10)]}, assessment(10)]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    docs = await runner.run([{"url": page}])
    assert visited == [page]
    assert loader.render.await_args.args[0] == page
    assert len(docs) == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    retry = next(e for e in events if e["type"] == "browser_retry")
    assert retry["reason"] == "HTTP 403"
    assert retry["attempted"] is True
    fetched = [e for e in events if e["type"] == "fetched"]
    assert [e["via"] for e in fetched] == ["http", "browser"]


async def test_blocked_pdf_is_not_retried(web, loader, tmp_path):
    """Documents cannot be rendered, so a blocked PDF fails cleanly."""
    bodies, _ = web
    pdf = "https://code.example/ord-1.pdf"
    bodies[pdf] = (403, {})
    caller = SimpleNamespace(
        call=AsyncMock(return_value={"links": [rating(0, 9)]})
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    assert await runner.run([{"url": pdf}]) == []
    loader.render.assert_not_awaited()
    assert runner.entries[pdf]["state"] == "failed"


async def test_search_uses_configured_engines(monkeypatch):
    """A run's search_engines setting overrides the default pair."""
    call = AsyncMock(return_value=[[[{"url": "https://ddg.example/a"}]]])
    monkeypatch.setattr("compass.web.search.search_all_se", call)
    out = await search_ordinance_candidates(
        ["q"], search_engines=["SerpAPIDuckDuckGoSearch"]
    )
    assert call.call_args.kwargs["search_engines"] == [
        "SerpAPIDuckDuckGoSearch"
    ]
    assert [c["url"] for c in out] == ["https://ddg.example/a"]


@pytest.mark.parametrize(
    ("kind", "accepted"),
    [("resolution", True), ("moratorium", True), ("summary", False)],
)
async def test_resolutions_and_moratoria_become_candidates(
    web, loader, tmp_path, kind, accepted
):
    """Adopted moratorium resolutions are passed on, summaries are not."""
    bodies, _ = web
    pdf = "https://county.gov/resolution-2026-134.pdf"
    bodies[pdf] = b"%PDF-resolution"
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[{"links": [rating(0, 10)]}, assessment(8, kind)]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    docs = await runner.run([{"url": pdf}])
    assert bool(docs) is accepted


async def test_candidates_carry_dates_and_later_ones_rank_first(
    web, loader, tmp_path
):
    """Equal scores rank by date; undated candidates keep no date."""
    bodies, _ = web
    urls = [
        "https://county.gov/resolution-2025-271.pdf",
        "https://county.gov/resolution-2026-134.pdf",
        "https://county.gov/undated.pdf",
    ]
    for url in urls:
        bodies[url] = url.encode()
    dates = [
        {"year": 2025, "month": 12, "day": 16, "evidence": "Adopted"},
        {"year": 2026, "month": 6, "day": 9, "evidence": "Adopted"},
        None,
    ]
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {"links": [rating(i, 10 - i) for i in range(3)]},
                *(assessment(9, "moratorium", date=d) for d in dates),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    docs = await runner.run([{"url": url} for url in urls])
    assert [doc.attrs["source"] for doc in docs] == [urls[1], urls[0], urls[2]]
    assert docs[0].attrs["date"] == (2026, 6, 9)
    assert "date" not in docs[2].attrs


async def test_stop_count_keeps_crawling_for_more_finds(web, loader, tmp_path):
    """With stop_count 2, one score-10 find does not end the crawl."""
    bodies, visited = web
    urls = [f"https://county.gov/{n}.pdf" for n in "abc"]
    for url in urls:
        bodies[url] = url.encode()
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {"links": [rating(i, 10 - i) for i in range(3)]},
                assessment(10),
                assessment(4),
                assessment(10),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller,
        loader,
        {"technology": "data centers"},
        events=_log(tmp_path),
        stop_count=2,
    )
    docs = await runner.run([{"url": url} for url in urls])
    assert visited == urls
    assert len(docs) == 3
    finished = json.loads(
        (tmp_path / "events.jsonl").read_text().splitlines()[-1]
    )
    assert finished["stop_reason"] == "found_document"


def test_adoption_date_pages_are_excerpted():
    """A signature page outside the sampled pages is still sent."""
    pages = ["General zoning code"] * 40
    pages[0] = "Data center moratorium"
    pages[20] = "PASSED AND ADOPTED this 9th day of June, 2026."
    pages[30] = "Printed January 3, 2026"
    excerpts = crawl._assessment_excerpts(PDFDocument(pages), "data centers")
    assert [x["page"] for x in excerpts] == [1, 2, 21, 39, 40]
    assert "9th day of June, 2026" in excerpts[2]["text"]


def test_html_adoption_dates_past_the_cutoff_are_excerpted():
    """Amendment history at the end of a long page is still sent."""
    text = (
        "Zoning text. " * 3000
        + "Amended by Ord. 26-14, effective March 2, 2026."
    )
    excerpts = crawl._assessment_excerpts(HTMLDocument([text]), "data centers")
    assert len(excerpts) == 2
    assert "March 2, 2026" in excerpts[1]["text"]


@pytest.mark.parametrize(
    "latest_date",
    [
        {"evidence": "No date stated"},
        {"evidence": "Ordained July 8, 2025", "year": 2025},
        None,
    ],
)
def test_assessment_tolerates_omitted_date_parts(latest_date):
    """Dropped date parts read as unknown instead of failing the page."""
    data = assessment(10)
    data["latest_date"] = latest_date
    parsed = crawl._DocumentAssessment.model_validate(data)
    assert parsed.latest_date.month is None
    schema = crawl._DocumentAssessment.model_json_schema()
    assert set(schema["$defs"]["_DocumentDate"]["required"]) == {
        "year",
        "month",
        "day",
        "evidence",
    }


@pytest.mark.parametrize(
    ("score", "kind", "status", "check_legal"),
    [
        (10, "ordinance", "adopted", False),
        (7, "code", "adopted", False),
        (6, "ordinance", "adopted", True),
        (8, "draft", "proposed", True),
        (8, "draft", "adopted", True),
        (9, "ordinance", "unknown", True),
    ],
)
async def test_adopted_legal_text_skips_the_legal_check(
    web, loader, tmp_path, score, kind, status, check_legal
):
    """The crawler's own assessment stands in for the legal-text check."""
    bodies, _ = web
    pdf = "https://county.gov/ordinance.pdf"
    bodies[pdf] = b"%PDF-ordinance"
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {"links": [rating(0, 10)]},
                assessment(score, kind, status=status),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    (doc,) = await runner.run([{"url": pdf}])
    assert doc.attrs["check_if_legal_doc"] is check_legal
    assert doc.attrs["check_correct_jurisdiction"] is True


async def test_blocked_download_records_the_whole_exchange(
    web, loader, tmp_path
):
    """A firewall challenge is kept: headers, body, and the browser try"""
    bodies, _ = web
    page = "https://county.gov/wp-admin/admin-ajax.php?task=file.download"
    challenge = (
        b"<html><title>You are being redirected...</title>"
        b"<noscript>Javascript is required.</noscript></html>"
    )
    headers = {
        "content-type": "text/html",
        "server": "Sucuri/Cloudproxy",
        "set-cookie": "sucuri_cloudproxy_uuid_1=abc; path=/",
    }
    bodies[page] = lambda request: httpx.Response(
        307, headers=headers, content=challenge, request=request
    )
    caller = SimpleNamespace(
        call=AsyncMock(return_value={"links": [rating(0, 10)]})
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    assert await runner.run([{"url": page}]) == []
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    http, browser = [e for e in events if e["type"] == "fetched"]
    assert http["response"]["status"] == 307
    assert http["response"]["headers"]["server"] == "Sucuri/Cloudproxy"
    assert http["response"]["headers"]["set-cookie"] == (
        "sucuri_cloudproxy_uuid_1=<redacted>"
    )
    assert http["request"]["method"] == "GET"
    assert "You are being redirected" in http["response"]["preview"]
    assert http["response"]["sha256"] == hashlib.sha256(challenge).hexdigest()
    assert browser["via"] == "browser"
    assert browser["empty"] is True
    (failed,) = [e for e in events if e["type"] == "visit_failed"]
    assert failed["url"] == canonical_url(page)
    assert failed["priority"] == 10
    assert failed["stage"] == "load"
    assert failed["tags"]["step"].endswith(".failed")
    assert failed["tags"]["document"] == canonical_url(page)
    branches = [
        (e["tags"]["step"].rsplit(".", 1)[1], e["outcome"])
        for e in events
        if e["type"] == "branch"
    ]
    assert branches == [("loaded", "no"), ("stop", "stop")]


async def test_duplicate_page_is_recorded_as_a_decision(web, loader, tmp_path):
    """Identical bytes at a second URL are skipped as a duplicate"""
    bodies, _ = web
    a, b = "https://city.gov/a.pdf", "https://city.gov/b.pdf"
    bodies[a] = bodies[b] = b"%PDF-same"
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {"links": [rating(0, 9), rating(1, 8)]},
                assessment(5),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    await runner.run([{"url": a}, {"url": b}])
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    new = [
        (e["tags"]["document"], e["outcome"])
        for e in events
        if e["type"] == "branch" and e["tags"]["step"].endswith(".new")
    ]
    assert new == [(a, "yes"), (b, "duplicate")]
    (skipped,) = [e for e in events if e["type"] == "skipped"]
    assert skipped["tags"]["step"].endswith(".skipped")


def _log(tmp_path):
    """Event log the tests read back from the temporary folder"""
    return EventLog(tmp_path / "events.jsonl")


def _events(tmp_path):
    return [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]


def _new_outcomes(events):
    return [
        e["outcome"]
        for e in events
        if e["type"] == "branch" and e["tags"]["step"].endswith(".new")
    ]


async def test_page_with_no_text_is_judged_by_what_it_frames(
    web, loader, tmp_path
):
    """A viewer page framing a PDF leads to the PDF, with its evidence"""
    bodies, visited = web
    topic = "https://county.gov/data-centers"
    viewer = "https://county.gov/resolution-2026-134/preview"
    pdf = "https://county.gov/files/resolution-2026-134.pdf"
    adoption = "On June 16, 2026 the Board approved Resolution No. 2026-134. "
    bodies[topic] = (
        f"<p>{adoption * 5}</p><a href='{viewer}'>Resolution No. 2026-134</a>"
    ).encode()
    bodies[viewer] = (
        "<html><head><title>2026-134</title></head><body>"
        f'<iframe src="{pdf}" title="2026-134"></iframe></body></html>'
    ).encode()
    bodies[pdf] = b"%PDF-1.7 resolution"
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {"links": [rating(0, 10)]},
                assessment(2, "summary", links=[rating(0, 10)]),
                # the viewer is scored as the resolution it shows
                assessment(10, "resolution", links=[rating(0, 10)]),
                assessment(10, "resolution"),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    docs = await runner.run([{"url": topic}])
    assert visited == [topic, viewer, pdf]
    # the viewer has no text, so only the PDF can be a candidate
    assert [doc.attrs["source"] for doc in docs] == [pdf]
    requests = [
        json.loads(c.kwargs["content"]) for c in caller.call.await_args_list
    ]
    viewer_request, pdf_request = requests[2], requests[3]
    assert "excerpts" not in viewer_request
    assert "<iframe" in viewer_request["html"]
    assert viewer_request["links"] == [
        {"id": 0, "title": "2026-134", "url": pdf}
    ]
    # the topic page's adoption statement reaches the PDF's assessment
    (parent,) = pdf_request["parents"]
    assert parent["parent_url"] == viewer
    assert "approved Resolution No. 2026-134" in parent["excerpts"][0]["text"]
    events = _events(tmp_path)
    assert _new_outcomes(events) == ["yes", "no text, has links", "yes"]
    assessed = [e for e in events if e["type"] == "assessed"]
    assert [e["has_text"] for e in assessed] == [True, False, True]
    assert [e["accepted_as_candidate"] for e in assessed] == [
        False,
        False,
        True,
    ]


async def test_browser_retry_that_renders_only_a_frame_is_read(
    web, loader, tmp_path
):
    """A blocked viewer page the browser renders without text is kept"""
    bodies, visited = web
    viewer = "https://county.gov/resolution-2026-134/preview"
    pdf = "https://county.gov/files/resolution-2026-134.pdf"
    bodies[viewer] = (307, {"server": "Sucuri/Cloudproxy"})
    bodies[pdf] = b"%PDF-1.7 resolution"
    wrapper = f'<html><body><iframe src="{pdf}" title="2026-134"></iframe>'
    # The real browser returns the wrapper's HTML; elm's
    # AsyncHTMLLoader.fetch would return an empty document for it
    loader.render.return_value = wrapper
    assert HTMLDocument([wrapper]).empty
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {"links": [rating(0, 10)]},
                assessment(0, "other", links=[rating(0, 10)]),
                assessment(10, "resolution"),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    docs = await runner.run([{"url": viewer}])
    assert visited == [viewer, pdf]
    assert [doc.attrs["source"] for doc in docs] == [pdf]
    events = _events(tmp_path)
    fetched = [
        (e["via"], e.get("empty")) for e in events if e["type"] == "fetched"
    ]
    assert fetched == [("http", None), ("browser", None), ("http", None)]
    assert _new_outcomes(events) == ["no text, has links", "yes"]


async def test_page_with_no_text_and_nothing_to_follow_is_skipped(
    web, loader, tmp_path
):
    """A blank page costs no assessment"""
    bodies, _ = web
    page = "https://city.gov/blank"
    bodies[page] = b"<html><body><div></div></body></html>"
    caller = SimpleNamespace(
        call=AsyncMock(return_value={"links": [rating(0, 10)]})
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    assert await runner.run([{"url": page}]) == []
    assert runner.entries[page]["state"] == "unreadable"
    assert caller.call.await_count == 1
    assert _new_outcomes(_events(tmp_path)) == ["unreadable"]


def test_pages_with_no_text_offer_documents_named_in_their_html():
    """A script or attribute naming a document is a lead without text"""
    html = (
        '<html><body><div id="viewer" data-file="/files/ord-12.pdf"></div>'
        '<script>var doc = {"url": "https:\\/\\/city.gov\\/files\\/'
        'ord-13.pdf"};</script></body></html>'
    )
    doc, links = crawl._html_document(html, "https://city.gov/viewer")
    assert not doc.text.strip()
    assert [link["url"] for link in links] == [
        "https://city.gov/files/ord-12.pdf",
        "https://city.gov/files/ord-13.pdf",
    ]
    # a page with text keeps only the links it shows
    text = f"<p>{'Data center ordinance text. ' * 20}</p>"
    doc, links = crawl._html_document(text + html, "https://city.gov/viewer")
    assert doc.text.strip()
    assert links == []


async def test_failed_browser_load_is_retried_once(web, loader, tmp_path):
    """A load that returns nothing gets a second, quicker attempt"""
    bodies, _ = web
    page = "https://code.example/enactments/Ord-2"
    bodies[page] = (403, {})
    loader.render.side_effect = [
        "",
        f"<p>{'Data center ordinance text. ' * 20}</p>",
    ]
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[{"links": [rating(0, 10)]}, assessment(10)]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    assert len(await runner.run([{"url": page}])) == 1
    states = [
        call.kwargs["load_state"] for call in loader.render.await_args_list
    ]
    assert states == ["networkidle", "domcontentloaded"]


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        (
            "https://drive.google.com/file/d/1LdY6_5ykJ70Fabeso/view?usp=sharing",
            "https://drive.google.com/uc?export=download&id=1LdY6_5ykJ70Fabeso",
        ),
        (
            "https://docs.google.com/file/d/AbC-123_x/edit",
            "https://drive.google.com/uc?export=download&id=AbC-123_x",
        ),
        (
            "https://drive.google.com/drive/folders/1AbC",
            "https://drive.google.com/drive/folders/1AbC",
        ),
        (
            "https://city.gov/file/d/123/view",
            "https://city.gov/file/d/123/view",
        ),
    ],
)
def test_google_drive_viewer_links_become_downloads(link, expected):
    """Drive's viewer shows only menus; the download is the file."""
    assert crawl._unwrap_document_viewer_url(link) == expected


async def test_google_drive_link_is_fetched_as_its_file(web, loader, tmp_path):
    """A township's Drive-hosted ordinance is read as the PDF it is"""
    bodies, visited = web
    page = "https://township.gov/ordinances-resolutions/"
    viewer = "https://drive.google.com/file/d/1LdY6/view?usp=sharing"
    download = "https://drive.google.com/uc?export=download&id=1LdY6"
    bodies[page] = (
        f"<p>{'Township ordinances and resolutions. ' * 10}</p>"
        f'<a href="{viewer}">Zoning Ordinance</a>'
    ).encode()
    bodies[download] = b"%PDF-1.7 zoning ordinance"
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {"links": [rating(0, 10)]},
                assessment(2, "summary", links=[rating(0, 10)]),
                assessment(10),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    docs = await runner.run([{"url": page}])
    assert visited == [page, download]
    assert [doc.attrs["source"] for doc in docs] == [download]


SKELETON = b"<html><body><div id='app'>Code Library</div></body></html>"
"""A script-built page as plain HTTP sees it: the same for every URL"""


async def test_pages_sharing_a_script_skeleton_are_rendered(
    web, loader, tmp_path
):
    """Identical skeletons are not duplicates until rendered as such."""
    bodies, _ = web
    home = "https://library.example/county"
    zoning = "https://library.example/county/code?nodeId=CH18ZO"
    bodies[home] = bodies[zoning] = SKELETON
    rendered = {
        home: "<p>Code index. " + "Chapter list. " * 20 + "</p>"
        "<a href='/county/code?nodeId=CH18ZO'>Chapter 18 Zoning</a>",
        zoning: "<p>Chapter 18. Data centers require a special use permit."
        + " Zoning text." * 20
        + "</p>",
    }
    loader.render.side_effect = lambda url, *_, **__: rendered[url]
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {"links": [rating(0, 9)]},
                assessment(2, "code", [rating(0, 9)]),
                assessment(9, "code"),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    docs = await runner.run([{"url": home}])
    assert runner.entries[zoning]["state"] == "assessed"
    assert docs[0].attrs["source"] == zoning
    assert "special use permit" in docs[0].text


async def test_pages_rendering_to_the_same_content_are_duplicates(
    web, loader, tmp_path
):
    """Two URLs for one rendered page cost one assessment."""
    bodies, _ = web
    a = "https://library.example/county/code"
    b = "https://library.example/county/code?view=print"
    bodies[a] = bodies[b] = SKELETON
    loader.render.return_value = (
        "<p>Data centers require a special use permit."
        + " Text." * 40
        + "</p>"
    )
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                {"links": [rating(0, 9), rating(1, 8)]},
                assessment(9, "code"),
            ]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    docs = await runner.run([{"url": a}, {"url": b}])
    assert len(docs) == 1
    assert caller.call.await_count == 2
    assert runner.entries[b]["state"] == "duplicate"
    assert runner.entries[b]["duplicate_of"] == runner.entries[a]["id"]


async def test_single_project_approvals_are_not_candidates(
    web, loader, tmp_path
):
    """A waiver for one data center is not the ordinance itself."""
    bodies, _ = web
    pdf = "https://city.legistar.com/View.ashx?ID=1"
    bodies[pdf] = b"%PDF-waiver"
    caller = SimpleNamespace(
        call=AsyncMock(
            side_effect=[{"links": [rating(0, 10)]}, assessment(3, "project")]
        )
    )
    runner = crawl.PriorityCrawler(
        caller, loader, {"technology": "data centers"}, events=_log(tmp_path)
    )
    assert await runner.run([{"url": pdf}]) == []
    assert "single project" in crawl._assessment_prompt(3)


def test_assessment_ignores_stray_fields():
    """An extra key beside the schema's fields keeps the assessment."""
    data = assessment(10)
    data["score"] = 10
    parsed = crawl._DocumentAssessment.model_validate(data)
    assert parsed.document_score == 10
    schema = crawl._DocumentAssessment.model_json_schema()
    assert schema["additionalProperties"] is False
