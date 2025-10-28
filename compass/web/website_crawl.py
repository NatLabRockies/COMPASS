"""Custom COMPASS website crawler

Much more simplistic than the Crawl4AI crawler, but designed to access
some links that Crawl4AI cannot (such as those behind a button
interface).
"""

import logging
import operator
from collections import Counter
from contextlib import AsyncExitStack
from urllib.parse import (
    urlparse,
    urlunparse,
    quote,
    unquote,
    parse_qsl,
    urlencode,
    urljoin,
)

from crawl4ai.models import Link as c4AILink
from bs4 import BeautifulSoup
from rebrowser_playwright.async_api import async_playwright
from rebrowser_playwright.async_api import Error as RBPlaywrightError
from playwright._impl._errors import Error as PlaywrightError  # noqa: PLC2701
from elm.web.utilities import pw_page
from elm.web.document import PDFDocument, HTMLDocument
from elm.web.file_loader import AsyncWebFileLoader
from elm.web.website_crawl import ELMLinkScorer, _SCORE_KEY  # noqa: PLC2701


logger = logging.getLogger(__name__)
_DEPTH_KEY = "web_crawl_depth"
_CLICKABLE_SELECTORS = [
    "button",  # "a", "p"
]
_BLACKLIST_SUBSTRINGS = [
    "facebook",
    "twitter",
    "linkedin",
    "instagram",
    "youtube",
    "instagram",
    "pinterest",  # cspell: disable-line
    "tiktok",  # cspell: disable-line
    "x.com",
    "snapchat",
    "reddit",
    "mailto:",
    "tel:",
    "javascript:",
    "login",
    "signup",
    "sign up",
    "signin",
    "sign in",
    "register",
    "subscribe",
    "donate",
    "shop",
    "cart",
    "careers",
    "event",
    "events",
    "calendar",
]
DOC_THRESHOLD = 5
"""Default max documents to collect before terminating COMPASS crawl"""


class COMPASSLinkScorer(ELMLinkScorer):
    """Custom URL scorer for COMPASS website crawling"""

    def _assign_value(self, text):
        """Score based on the presence of keywords in link text"""
        score = 0
        text = text.casefold().replace("plant", "")
        for kw, kw_score in self.keyword_points.items():
            if kw in text:
                score += kw_score
        return score


class _Link(c4AILink):
    """Crawl4AI Link subclass with a few utilities"""

    def __hash__(self):
        return hash(self.href.casefold())

    def __repr__(self):
        return (
            f"Link(title={self.title!r}, href={self.href!r}, "
            f"base_domain={self.base_domain!r})"
        )

    def __str__(self):
        return f"{self.title} ({self.href})"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.href.casefold() == other.casefold()

        if not isinstance(other, c4AILink):
            return NotImplemented
        return self.href.casefold() == other.href.casefold()

    @property
    def consistent_domain(self):
        """bool: ``True`` if the link is from the base domain"""
        return self.base_domain.casefold() in self.href.casefold()

    @property
    def resembles_pdf(self):
        """bool: ``True`` if the link has "pdf" in title or href"""
        return "pdf" in self.title.casefold() or "pdf" in self.href.casefold()


class COMPASSCrawler:
    """A simple website crawler to search for ordinance documents"""

    def __init__(
        self,
        validator,
        url_scorer,
        file_loader_kwargs=None,
        already_visited=None,
        num_link_scores_to_check_per_page=4,
        max_pages=100,
        browser_semaphore=None,
    ):
        """

        Parameters
        ----------
        validator : callable
            An async callable that takes a document instance (containing
            the text from a PDF or a webpage) and returns a boolean
            indicating whether the text passes the validation check.
            This is used to determine whether or not to keep (i.e.
            return) the document.
        url_scorer : callable
            An async callable that takes a list of dictionaries
            containing URL information and assigns each dictionary a
            `score` key representing the score for that URL. The input
            URL dictionaries will each have at least one key: "href".
            This key will contain the URL of the link. The dictionary
            may also have other attributes such as "title", which
            contains the link title text.
        file_loader_kwargs : dict, optional
            Additional keyword-value argument pairs to pass to the
            :class:`~elm.web.file_loader.AsyncWebFileLoader` class. If
            this dictionary contains the ``pw_launch_kwargs`` key, it's
            value (assumes to be another dictionary) will be used to
            initialize the playwright instances used for the crawl.
            By default, ``None``.
        already_visited : set, optional
            A set of URLs (either strings or ``Link`` objects)
            that have already been visited. This is used to avoid
            re-visiting links that have already been checked.
            By default, ``None``.
        num_link_scores_to_check_per_page : int, default=3
            Number of top unique-scoring links per page to use for
            recursive crawling. This helps the crawl stay focused on the
            most likely links to contain documents of interest.
        max_pages : int, default=100
            Maximum number of pages to crawl before terminating,
            regardless of whether the document was found or not.
            By default, ``100``.
        browser_semaphore : :class:`asyncio.Semaphore`, optional
            Semaphore instance that can be used to limit the number of
            playwright browsers open concurrently. If ``None``, no
            limits are applied. By default, ``None``.
        """
        self.validator = validator
        self.url_scorer = url_scorer
        self.num_scores_to_check_per_page = num_link_scores_to_check_per_page
        self.checked_previously = already_visited or set()
        self.max_pages = max_pages

        file_loader_kwargs = file_loader_kwargs or {}
        flk = {"verify_ssl": False}
        flk.update(file_loader_kwargs or {})
        self.afl = AsyncWebFileLoader(**flk)
        self.pw_launch_kwargs = (
            file_loader_kwargs.get("pw_launch_kwargs") or {}
        )
        self.browser_semaphore = (
            AsyncExitStack()
            if browser_semaphore is None
            else browser_semaphore
        )

        self._out_docs = []
        self._already_visited = {}
        self._should_stop = None

    async def run(
        self, base_url, termination_callback=None, on_new_page_visit_hook=None
    ):
        """Run the COMPASS website crawler

        Parameters
        ----------
        base_url : str
            URL of the website to start crawling from.
        termination_callback : callable, optional
            An async callable that takes a list of documents and returns
            a boolean indicating whether to stop crawling. If ``None``,
            the crawl will simply terminates when :obj:`DOC_THRESHOLD`
            number of documents have been found. By default, ``None``.
        on_new_page_visit_hook : callable, optional
            An async callable that is called every time a new page is
            found during the crawl. The callable should accept a single
            argument, which is the page ``Link`` instance.
            If ``None``, no additional processing is done on new pages.
            By default, ``None``.

        Returns
        -------
        list
            List of document instances that passed the validation
            check. Each document contains the text from a PDF and has an
            attribute `source` that contains the URL of the document.
            This list may be empty if no documents are found.
        """
        self._should_stop = termination_callback or _default_found_enough_docs
        await self._run(
            base_url, on_new_page_visit_hook=on_new_page_visit_hook
        )
        self._should_stop = None

        self._log_crawl_stats()

        if self._out_docs:
            self._out_docs.sort(key=lambda x: -1 * x.attrs[_SCORE_KEY])

        return self._out_docs

    async def _run(
        self,
        base_url,
        link=None,
        depth=0,
        score=0,
        on_new_page_visit_hook=None,
    ):
        """Recursive web crawl function"""
        if link is None:
            base_url, link = self._reset_crawl(base_url)

        if link in self._already_visited:
            return

        if on_new_page_visit_hook:
            await on_new_page_visit_hook(link)

        self._already_visited[link] = (depth, score)
        logger.trace("self._already_visited=%r", self._already_visited)

        if await self._website_link_is_doc(link, depth, score):
            return

        num_urls_checked_on_this_page = 0
        curr_url_score = None
        for next_link in await self._get_links_from_page(link, base_url):
            prev_len = len(self._out_docs)
            await self._run(
                base_url,
                link=_Link(
                    title=next_link["title"],
                    href=next_link["href"],
                    base_domain=base_url,
                ),
                depth=depth + 1,
                score=next_link["score"],
                on_new_page_visit_hook=on_new_page_visit_hook,
            )

            doc_was_just_found = (  # fmt: off
                len(self._out_docs) == (prev_len + 1)
                and (
                    self._out_docs[-1].attrs.get(_DEPTH_KEY, -1) == (depth + 1)
                )
            )
            if doc_was_just_found:
                if await self.validator(self._out_docs[-1]):
                    logger.debug("    - Document passed validation check!")
                else:
                    self._out_docs = self._out_docs[:-1]
            elif (
                not link.resembles_pdf and curr_url_score != next_link["score"]
            ):
                logger.trace(
                    "Finished checking score %d at depth %d. Next score: %d",
                    curr_url_score or -1,
                    depth,
                    next_link["score"],
                )
                num_urls_checked_on_this_page += 1
                curr_url_score = next_link["score"]

            if await self._should_terminate_crawl(
                num_urls_checked_on_this_page, link
            ):
                break

        return

    def _reset_crawl(self, base_url):
        """Reset crawl state and initialize crawling link"""
        self._out_docs = []
        self._already_visited = {}
        base_url = _sanitize_url(base_url)
        return base_url, _Link(
            title="Landing Page",
            href=_sanitize_url(urljoin(base_url, base_url.split(" ")[0])),
            base_domain=base_url,
        )

    async def _website_link_is_doc(self, link, depth, score):
        """Check if website link contains doc"""
        if link in self.checked_previously and link.consistent_domain:
            # Don't re-check pages on main website
            return False

        if await self._website_link_is_pdf(link, depth, score):
            return True

        # at this point the page is NOT a PDF. However, it could still
        # just be a normal webpage on the main domain that we haven't
        # visited before. In that case, just return False
        if not link.consistent_domain:
            return False

        # now we are on an external page that we either have not visited
        # before or that we have seen but determined is NOT a PDF file
        return await self._website_link_as_html_doc(link, depth, score)

    async def _website_link_is_pdf(self, link, depth, score):
        """Fetch page content; keep only PDFs"""
        logger.debug("Loading Link: %s", link)

        try:
            doc = await self.afl.fetch(link.href)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            msg = (
                "Encountered error of type %r while trying to fetch "
                "content from %s"
            )
            err_type = type(e)
            logger.exception(msg, err_type, link)
            return False

        if isinstance(doc, PDFDocument):
            logger.debug("    - Found PDF!")
            doc.attrs[_DEPTH_KEY] = depth
            doc.attrs[_SCORE_KEY] = score
            self._out_docs.append(doc)
            return True

        return False

    async def _website_link_as_html_doc(self, link, depth, score):
        """Fetch page content as HTML doc"""
        logger.debug("Loading Link as HTML: %s", link)
        html_text = await self._get_text_no_err(link.href)

        attrs = {_DEPTH_KEY: depth, _SCORE_KEY: score}
        doc = HTMLDocument([html_text], attrs=attrs)
        self._out_docs.append(doc)
        return True

    async def _get_links_from_page(self, link, base_url):
        """Get all links from a page sorted by relevance score"""
        if not link.consistent_domain:
            logger.debug("Detected new domain, stopping link discovery")
            return []

        html_text = await self._get_text_no_err(link.href)
        page_links = []
        if html_text:
            page_links = _extract_links_from_html(html_text, base_url=base_url)
            page_links = await self.url_scorer(
                [dict(link) for link in page_links]
            )
            page_links = sorted(
                page_links, key=operator.itemgetter("score"), reverse=True
            )
        _debug_info_on_links(page_links)
        return page_links

    async def _get_text_no_err(self, url):
        """Get all text from a page; return empty string if pw error"""
        try:
            text = await self._get_text(url)
        except (PlaywrightError, RBPlaywrightError):
            text = ""

        return text

    async def _get_text(self, url):
        """Get all html text from a page"""
        all_text = []
        pw_page_kwargs = {
            "intercept_routes": True,
            "ignore_https_errors": True,
            "timeout": 60_0000,
        }
        async with async_playwright() as p, self.browser_semaphore:
            browser = await p.chromium.launch(**self.pw_launch_kwargs)
            async with pw_page(browser, **pw_page_kwargs) as page:
                await page.goto(url)
                await page.wait_for_load_state("networkidle", timeout=60_000)

                all_text.append(await page.content())
                all_text += await _get_text_from_all_locators(page)

        return "\n".join(all_text)

    async def _should_terminate_crawl(
        self, num_urls_checked_on_this_page, link
    ):
        """Check if crawl should terminate"""
        if num_urls_checked_on_this_page >= self.num_scores_to_check_per_page:
            logger.debug(
                "Already checked %d unique link scores from %s",
                self.num_scores_to_check_per_page,
                link.href,
            )
            return True

        if await self._should_stop(self._out_docs):
            logger.debug("Exiting crawl early due to user condition")
            return True

        if len(self._already_visited) >= self.max_pages:
            logger.debug("    - Too many links visited, stopping recursion")
            return True

        logger.trace(
            "Only checked %d pages, continuing crawl...",
            len(self._already_visited),
        )
        return False

    def _log_crawl_stats(self):
        """Log statistics about crawled pages and depths"""
        logger.info("Crawled %d pages", len(self._already_visited))
        logger.info("Found %d potential documents", len(self._out_docs))
        logger.debug("Average score: %.2f", self._compute_avg_link_score())

        logger.debug("Pages crawled by depth:")
        for depth, count in sorted(self._crawl_depth_counts().items()):
            logger.debug("  Depth %d: %d pages", depth, count)

    def _compute_avg_link_score(self):
        """Compute the average score of the crawled results"""
        return sum(
            score for __, score in self._already_visited.values()
        ) / len(self._already_visited)

    def _crawl_depth_counts(self):
        """Compute number of pages per depth"""
        depth_counts = Counter()
        depth_counts.update([d for d, __ in self._already_visited.values()])
        return depth_counts


async def _default_found_enough_docs(out_docs):  # noqa: RUF029
    """Check if a predetermined # of documents has been found

    The number to check is set by the module-level constant
    :obj:`DOC_THRESHOLD`.
    """
    return len(out_docs) >= DOC_THRESHOLD


def _debug_info_on_links(links):
    """Send debug info on links to logger"""
    num_links = len(links)
    if num_links <= 0:
        logger.debug("Found no links!")
        return

    logger.debug("Found %d links:", len(links))

    for link in links[:3]:
        logger.debug(
            "    - %d: %s (%s)", link["score"], link["title"], link["href"]
        )
    if num_links > 3:  # noqa: PLR2004
        logger.debug("    ...")


def _sanitize_url(url):
    """Fix common URL issues

    - Encode spaces and unsafe characters in the path
    - Encode query parameters safely
    - Leave existing percent-encoding intact
    """
    parsed = urlparse(url)

    safe_path = quote(unquote(parsed.path), safe="/")

    query_params = parse_qsl(parsed.query, keep_blank_values=True)
    safe_query = urlencode(query_params, doseq=True)  # cspell: disable-line

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            safe_path,
            parsed.params,
            safe_query,
            parsed.fragment,
        )
    )


def _extract_links_from_html(text, base_url):
    """Parse HTML and extract all links"""
    soup = BeautifulSoup(text, "html.parser")
    links = [
        (a.get_text().strip(), a["href"])
        for a in soup.find_all("a", href=True)
    ]

    return {
        _Link(
            title=title,
            href=_sanitize_url(urljoin(base_url, path)),
            base_domain=base_url,
        )
        for (title, path) in links
        if title
        and path
        and not any(
            substr in title.lower() for substr in _BLACKLIST_SUBSTRINGS
        )
        and not any(substr in path.lower() for substr in _BLACKLIST_SUBSTRINGS)
    }


async def _get_text_from_all_locators(page):
    """Go through locators on page and get text behind them"""
    all_text = []
    for selector in _CLICKABLE_SELECTORS:
        logger.trace("Checking selector %r", selector)
        locators = page.locator(selector)
        locator_count = await locators.count()
        logger.trace("    - Found %d instances", locator_count)
        for index in range(locator_count):
            try:
                text = await _get_locator_text(locators, index, page)
            except (PlaywrightError, RBPlaywrightError):
                continue
            if text:
                all_text.append(text)
    return all_text


async def _get_locator_text(locators, index, page):
    """Get text after clicking on one of the page locators"""
    locator = locators.nth(index)

    if not await locator.is_visible():
        return None

    if not await locator.is_enabled():
        return None

    await locator.click(timeout=10_000)
    return await page.content()
