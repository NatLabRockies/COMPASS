"""COMPASS web crawling tests"""

import asyncio
import logging
import types
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from crawl4ai.models import Link as TestLink

from compass.web import website_crawl
from compass.web.website_crawl import (
    COMPASSCrawler,
    COMPASSLinkScorer,
    DOC_THRESHOLD,
    _DEPTH_KEY,
    _SCORE_KEY,
    _Link,
    _debug_info_on_links,
    _default_found_enough_docs,
    _extract_links_from_html,
    _get_locator_text,
    _get_text_from_all_locators,
    _sanitize_url,
)


class StubLocator:
    """Simple locator stub that mimics Playwright locator behavior"""

    def __init__(self, *, visible=True, enabled=True, content=None, exc=None):
        self.visible = visible
        self.enabled = enabled
        self.content = content
        self.exc = exc
        self.page = None
        self.clicks = 0

    async def is_visible(self):
        return self.visible

    async def is_enabled(self):
        return self.enabled

    async def click(self, timeout=10_000):
        self.clicks += 1
        if self.exc:
            raise self.exc
        if self.content is not None and self.page is not None:
            self.page.set_html(self.content)


class StubLocators:
    """Container for Playwright locator collections"""

    def __init__(self, page, locators):
        self._page = page
        self._locators = list(locators)

    async def count(self):
        return len(self._locators)

    def nth(self, index):
        locator = self._locators[index]
        locator.page = self._page
        return locator


class StubPage:
    """Stub Playwright page for deterministic content capture"""

    def __init__(self, html, locator_map=None):
        self._html = html
        self._locator_map = locator_map or {}
        self.visited = []

    async def goto(self, url):
        self.visited.append(url)

    async def wait_for_load_state(self, *_args, **_kwargs):
        return None

    async def content(self):
        return self._html

    def set_html(self, html):
        self._html = html

    def locator(self, selector):
        locators = self._locator_map.get(selector, [])
        return StubLocators(self, locators)


@pytest.fixture
def crawler_setup(monkeypatch):
    """Provide a COMPASS crawler with deterministic dependencies"""

    class DummyPDFDocument:
        def __init__(self, text, attrs=None):
            self.text = text
            self.attrs = attrs or {}
            self.source = self.attrs.get("source", "pdf")

    class DummyHTMLDocument:
        def __init__(self, parts, attrs=None):
            self.parts = list(parts)
            self.text = "\n".join(parts)
            self.attrs = attrs or {}
            self.source = self.attrs.get("source", "html")

    loader_docs = {}

    class DummyLoader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.loader_docs = loader_docs
            self.fetch_calls = []

        async def fetch(self, url):
            self.fetch_calls.append(url)
            entry = self.loader_docs.get(url)
            if isinstance(entry, Exception):
                raise entry
            if entry is not None:
                return entry
            return DummyHTMLDocument(
                [f"<html>{url}</html>"], attrs={"source": url}
            )

    monkeypatch.setattr(website_crawl, "PDFDocument", DummyPDFDocument)
    monkeypatch.setattr(website_crawl, "HTMLDocument", DummyHTMLDocument)
    monkeypatch.setattr(website_crawl, "AsyncWebFileLoader", DummyLoader)

    async def validator(doc):
        await asyncio.sleep(0)
        return "keep" in getattr(doc, "text", "")

    async def scorer(links):
        await asyncio.sleep(0)
        for idx, info in enumerate(links):
            info["score"] = 100 - idx
        return links

    crawler = COMPASSCrawler(
        validator=validator,
        url_scorer=scorer,
        num_link_scores_to_check_per_page=1,
        max_pages=4,
    )
    return {
        "crawler": crawler,
        "loader_docs": loader_docs,
        "pdf_cls": DummyPDFDocument,
        "html_cls": DummyHTMLDocument,
    }


@pytest.fixture(scope="module")
def compass_logger():
    """Provide compass logger with DEBUG_TO_FILE level for tests"""
    logger = logging.getLogger("compass")
    prev_level = logger.level
    logger.setLevel("DEBUG_TO_FILE")
    try:
        yield logger
    finally:
        logger.setLevel(prev_level)


def test_link_equality():
    """Test equality of Link instances"""

    assert _Link() == _Link()
    assert _Link() == _Link(title="test")
    assert _Link() == _Link(text="test")
    assert _Link() == _Link(base_domain="test")
    assert _Link() != _Link(href="test")

    link1 = TestLink(title="test", href="http://example.com/test")
    link2 = _Link(title="Test", href="http://example.com/test")

    assert link1 == link2
    assert link2 == "http://example.com/test"

    assert link2 != "http://example.com/test2"

    assert link2 in {"http://example.com/test", "http://example.com/test2"}
    assert link2 not in {
        "http://example.com/test2",
        "http://example.com/test3",
    }

    assert link2 == "http://example.com/test"


def test_link_consistent_domain():
    """Test `Link.consistent_domain` property"""

    assert _Link().consistent_domain
    assert not _Link(base_domain="example.com").consistent_domain
    assert _Link(
        href="example.com/test", base_domain="example.com"
    ).consistent_domain


def test_link_resembles_pdf():
    """Test `Link.resembles_pdf` property"""

    assert not _Link().resembles_pdf
    assert _Link(title="example.pdf").resembles_pdf
    assert _Link(href="example.pdf").resembles_pdf
    assert not _Link(base_domain="example.pdf").resembles_pdf


def test_link_hash_and_repr():
    """Ensure hash, repr, and str outputs are informative"""

    link = _Link(
        title="Example",
        href="https://example.com/path",
        base_domain="https://example.com",
    )
    assert isinstance(hash(link), int)
    assert "Example" in repr(link)
    assert "https://example.com/path" in str(link)


def test_compass_link_scorer_assign_value():
    """Validate keyword scoring logic unique to COMPASS scorer"""

    scorer = object.__new__(COMPASSLinkScorer)
    scorer.keyword_points = {"solar": 3, "energy": 5}

    assert scorer._assign_value("Solar energy plant") == 8
    assert scorer._assign_value("Hydro only") == 0


def test_sanitize_url_handles_spaces_and_queries():
    """Verify URL sanitization for paths and query strings"""

    sanitized = _sanitize_url("https://example.com/some path/?foo=bar baz")
    assert " " not in sanitized
    assert "%20" in sanitized


def test_extract_links_from_html_filters_blacklist():
    """Ensure blacklist filtering removes social links"""

    html = """
    <a href="/keep">Keep Link</a>
    <a href="https://facebook.com/page">Facebook</a>
    <a href="https://example.com/ok.pdf">PDF Title</a>
    """
    links = _extract_links_from_html(html, base_url="https://example.com")
    test_refs = {link.href for link in links}

    assert "https://example.com/keep" in test_refs
    assert all("facebook" not in link.href for link in links)
    assert "https://example.com/ok.pdf" in test_refs


def test_debug_info_on_links_logs_expected(
    compass_logger, assert_message_was_logged
):
    """Check debug logging for link collections"""

    _debug_info_on_links([])
    assert_message_was_logged(
        "Found no links", log_level="DEBUG", clear_records=True
    )

    links = [
        {"score": idx, "title": f"Doc {idx}", "href": f"https://e/{idx}"}
        for idx in range(5)
    ]
    _debug_info_on_links(links)
    assert_message_was_logged("Found 5 links", log_level="DEBUG")
    assert_message_was_logged("Doc 0", log_level="DEBUG")


@pytest.mark.asyncio
async def test_default_found_enough_docs_threshold():
    """Validate default termination threshold logic"""

    docs = [None] * DOC_THRESHOLD
    assert await _default_found_enough_docs(docs)
    assert not await _default_found_enough_docs(docs[:-1])


@pytest.mark.asyncio
async def test_get_locator_text_returns_none_when_not_visible():
    """Locator text fetch skips invisible elements"""

    page = StubPage("<html></html>")
    locators = StubLocators(page, [StubLocator(visible=False)])
    assert await _get_locator_text(locators, 0, page) is None


@pytest.mark.asyncio
async def test_get_locator_text_returns_none_when_not_enabled():
    """Locator text fetch skips disabled elements"""

    page = StubPage("<html></html>")
    locators = StubLocators(page, [StubLocator(enabled=False)])
    assert await _get_locator_text(locators, 0, page) is None


@pytest.mark.asyncio
async def test_get_locator_text_returns_content_on_click():
    """Locator text fetch returns updated page content post-click"""

    updated_html = "<html>clicked</html>"
    page = StubPage("<html>original</html>")
    locators = StubLocators(page, [StubLocator(content=updated_html)])
    assert await _get_locator_text(locators, 0, page) == updated_html


@pytest.mark.asyncio
async def test_get_text_from_all_locators_collects_text():
    """Collect text produced by clicking configured selectors"""

    updated_html = "<html>after</html>"
    page = StubPage(
        "<html>before</html>",
        locator_map={
            "button": [StubLocator(content=updated_html)],
        },
    )
    assert await _get_text_from_all_locators(page) == [updated_html]


@pytest.mark.asyncio
async def test_get_text_from_all_locators_ignores_errors():
    """Ensure Playwright errors are swallowed during locator walks"""

    page = StubPage(
        "<html>start</html>",
        locator_map={
            "button": [
                StubLocator(
                    exc=website_crawl.PlaywrightError("error"),
                )
            ]
        },
    )
    assert await _get_text_from_all_locators(page) == []


def test_reset_crawl_sanitizes_and_initializes(crawler_setup):
    """Resetting the crawler should clear state and sanitize URLs"""

    crawler = crawler_setup["crawler"]
    base_url, start_link = crawler._reset_crawl(
        "https://example.com/path with space/"
    )
    assert " " not in base_url
    assert start_link.href.startswith("https://example.com")
    assert crawler._out_docs == []
    assert crawler._already_visited == {}


@pytest.mark.asyncio
async def test_website_link_is_doc_skips_pre_checked(crawler_setup):
    """Links flagged as previously checked are skipped"""

    crawler = crawler_setup["crawler"]
    link = _Link(
        title="Checked",
        href="https://example.com/page",
        base_domain="https://example.com",
    )
    crawler.checked_previously.add(link)
    assert not await crawler._website_link_is_doc(link, 0, 0)


@pytest.mark.asyncio
async def test_website_link_is_doc_external_returns_false(crawler_setup):
    """External domains should return false and not create docs"""

    crawler = crawler_setup["crawler"]
    link = _Link(
        title="External",
        href="https://other.com/file",
        base_domain="https://example.com",
    )
    assert not await crawler._website_link_is_doc(link, 0, 0)


@pytest.mark.asyncio
async def test_website_link_is_pdf_adds_document(crawler_setup):
    """PDF links should be fetched and appended to output docs"""

    crawler = crawler_setup["crawler"]
    loader_docs = crawler_setup["loader_docs"]
    pdf_cls = crawler_setup["pdf_cls"]

    link = _Link(
        title="PDF",
        href="https://example.com/doc.pdf",
        base_domain="https://example.com",
    )
    loader_docs[link.href] = pdf_cls(
        "keep document", attrs={"source": link.href}
    )

    assert await crawler._website_link_is_pdf(link, depth=1, score=7)
    assert crawler._out_docs[-1].attrs[_DEPTH_KEY] == 1
    assert crawler._out_docs[-1].attrs[_SCORE_KEY] == 7


@pytest.mark.asyncio
async def test_website_link_is_pdf_handles_exception(crawler_setup):
    """Errors during PDF fetch should be logged and ignored"""

    crawler = crawler_setup["crawler"]
    loader_docs = crawler_setup["loader_docs"]
    link = _Link(
        title="Bad PDF",
        href="https://example.com/bad.pdf",
        base_domain="https://example.com",
    )
    loader_docs[link.href] = RuntimeError("error")

    assert not await crawler._website_link_is_pdf(link, depth=0, score=0)
    assert all(
        doc.attrs.get("source") != link.href for doc in crawler._out_docs
    )


@pytest.mark.asyncio
async def test_website_link_as_html_doc_adds_document(
    crawler_setup, monkeypatch
):
    """HTML pages should be converted into HTML document objects"""

    crawler = crawler_setup["crawler"]

    async def fake_get_text(self, url):
        await asyncio.sleep(0)
        return "<html>keep content</html>"

    monkeypatch.setattr(
        crawler,
        "_get_text_no_err",
        types.MethodType(fake_get_text, crawler),
    )

    link = _Link(
        title="HTML",
        href="https://example.com/page",
        base_domain="https://example.com",
    )
    assert await crawler._website_link_as_html_doc(link, depth=2, score=9)
    doc = crawler._out_docs[-1]
    assert doc.attrs[_DEPTH_KEY] == 2
    assert doc.attrs[_SCORE_KEY] == 9
    assert "keep" in doc.text


@pytest.mark.asyncio
async def test_get_links_from_page_skips_inconsistent_domain(
    crawler_setup, monkeypatch
):
    """No links should be fetched when the domain changes"""

    crawler = crawler_setup["crawler"]

    async def fail_get_text(self, url):
        await asyncio.sleep(0)
        raise AssertionError("Should not fetch external domains")

    monkeypatch.setattr(
        crawler,
        "_get_text_no_err",
        types.MethodType(fail_get_text, crawler),
    )

    link = _Link(
        title="External",
        href="https://other.com/page",
        base_domain="https://example.com",
    )
    assert (
        await crawler._get_links_from_page(link, "https://example.com") == []
    )


@pytest.mark.asyncio
async def test_get_links_from_page_returns_sorted_scores(
    crawler_setup, monkeypatch
):
    """Links should be scored and returned in descending order"""

    crawler = crawler_setup["crawler"]

    async def fake_get_text(self, url):
        await asyncio.sleep(0)
        return """
        <a href="/keep.pdf">Keep</a>
        <a href="/other.pdf">Other</a>
        <a href="/normal">Normal</a>
        """

    async def scorer(urls):
        await asyncio.sleep(0)
        score_map = {
            "https://example.com/keep.pdf": 30,
            "https://example.com/other.pdf": 20,
            "https://example.com/normal": 10,
        }
        for info in urls:
            info["score"] = score_map[info["href"]]
        return urls

    monkeypatch.setattr(
        crawler,
        "_get_text_no_err",
        types.MethodType(fake_get_text, crawler),
    )
    crawler.url_scorer = scorer

    link = _Link(
        title="Base",
        href="https://example.com/index",
        base_domain="https://example.com",
    )
    results = await crawler._get_links_from_page(link, "https://example.com")
    assert [item["score"] for item in results] == [30, 20, 10]
    assert results[0]["title"] == "Keep"


@pytest.mark.asyncio
async def test_get_text_no_err_handles_playwright_error(
    crawler_setup, monkeypatch
):
    """Playwright errors should yield empty string safely"""

    crawler = crawler_setup["crawler"]

    async def raise_error(self, url):
        await asyncio.sleep(0)
        raise website_crawl.PlaywrightError("error")

    monkeypatch.setattr(
        crawler,
        "_get_text",
        types.MethodType(raise_error, crawler),
    )
    assert not await crawler._get_text_no_err("https://example.com")


@pytest.mark.asyncio
async def test_get_text_uses_playwright_and_collects_content(
    crawler_setup, monkeypatch
):
    """Ensure playwright usage collects page content and locator output"""

    crawler = crawler_setup["crawler"]
    page = StubPage("<html>body</html>")

    async def fake_text_from_locators(page):
        await asyncio.sleep(0)
        return ["clicked"]

    monkeypatch.setattr(
        website_crawl, "_get_text_from_all_locators", fake_text_from_locators
    )

    class StubBrowser:
        def __init__(self, page):
            self.page = page

    class StubChromium:
        def __init__(self, page):
            self._page = page

        async def launch(self, **_kwargs):
            return StubBrowser(self._page)

    class StubPlaywright:
        def __init__(self, page):
            self.chromium = StubChromium(page)

    @asynccontextmanager
    async def fake_async_playwright():
        await asyncio.sleep(0)
        yield StubPlaywright(page)

    @asynccontextmanager
    async def fake_pw_page(browser, **_kwargs):
        await asyncio.sleep(0)
        yield page

    monkeypatch.setattr(
        website_crawl, "async_playwright", fake_async_playwright
    )
    monkeypatch.setattr(website_crawl, "pw_page", fake_pw_page)

    result = await crawler._get_text("https://example.com")
    assert result == "<html>body</html>\nclicked"
    assert "https://example.com" in page.visited


@pytest.mark.asyncio
async def test_should_terminate_crawl_conditions(crawler_setup):
    """Cover termination branches for score limits, callback, and max pages"""

    crawler = crawler_setup["crawler"]
    test_link = _Link(
        title="Base",
        href="https://example.com/base",
        base_domain="https://example.com",
    )

    assert await crawler._should_terminate_crawl(1, test_link)

    async def stop_true(out_docs):
        await asyncio.sleep(0)
        return True

    crawler._should_stop = stop_true
    crawler.num_scores_to_check_per_page = 99
    assert await crawler._should_terminate_crawl(0, test_link)

    async def stop_false(out_docs):
        await asyncio.sleep(0)
        return False

    crawler._should_stop = stop_false
    crawler.max_pages = 1
    crawler._already_visited = {test_link: (0, 0)}
    assert await crawler._should_terminate_crawl(0, test_link)

    crawler.max_pages = 5
    crawler._already_visited = {test_link: (0, 10)}
    assert not await crawler._should_terminate_crawl(0, test_link)


def test_compute_avg_score_and_depth_counts(crawler_setup):
    """Average score and depth counts reflect visited pages"""

    crawler = crawler_setup["crawler"]
    link_a = _Link(
        title="A",
        href="https://example.com/a",
        base_domain="https://example.com",
    )
    link_b = _Link(
        title="B",
        href="https://example.com/b",
        base_domain="https://example.com",
    )
    crawler._already_visited = {link_a: (0, 10), link_b: (2, 30)}

    assert crawler._compute_avg_link_score() == 20
    counts = crawler._crawl_depth_counts()
    assert counts[0] == 1
    assert counts[2] == 1


def test_log_crawl_stats_emits_messages(
    crawler_setup, compass_logger, assert_message_was_logged
):
    """Ensure crawl statistics are logged"""

    crawler = crawler_setup["crawler"]
    pdf_cls = crawler_setup["pdf_cls"]
    doc = pdf_cls("keep", attrs={_SCORE_KEY: 42, _DEPTH_KEY: 1})
    crawler._out_docs = [doc]
    link = _Link(
        title="A",
        href="https://example.com/a",
        base_domain="https://example.com",
    )
    crawler._already_visited = {link: (0, 42)}

    crawler._log_crawl_stats()
    assert_message_was_logged("Crawled 1 pages", log_level="INFO")
    assert_message_was_logged("Found 1 potential documents", log_level="INFO")


@pytest.mark.asyncio
async def test_run_sorts_documents_and_resets_state(
    crawler_setup, monkeypatch
):
    """`run` should delegate to `_run`, sort docs, and reset stop callback"""

    crawler = crawler_setup["crawler"]
    pdf_doc = crawler_setup["pdf_cls"](
        "keep pdf",
        attrs={_SCORE_KEY: 10, _DEPTH_KEY: 1, "source": "pdf"},
    )
    html_doc = crawler_setup["html_cls"](
        ["keep html"],
        attrs={_SCORE_KEY: 25, _DEPTH_KEY: 2, "source": "html"},
    )

    async def fake_run(
        self,
        base_url,
        link=None,
        depth=0,
        score=0,
        on_new_page_visit_hook=None,
    ):
        await asyncio.sleep(0)
        self._out_docs = [pdf_doc, html_doc]
        self._already_visited = {
            _Link(title="Landing", href=base_url, base_domain=base_url): (0, 5)
        }

    monkeypatch.setattr(crawler, "_run", types.MethodType(fake_run, crawler))

    async def stopper(docs):
        await asyncio.sleep(0)
        return False

    docs = await crawler.run(
        "https://example.com", termination_callback=stopper
    )

    assert docs == [html_doc, pdf_doc]
    assert crawler._should_stop is None


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
