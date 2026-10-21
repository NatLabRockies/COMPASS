"""Exercise container assets without root or external networking"""

import os
import asyncio
import socket
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from threading import Thread

from compass.exceptions import COMPASSRuntimeError


_HTML = """<!doctype html>
<html><head><title>COMPASS checks</title></head><body>
<h1>COMPASS fixture</h1>
<p>The county ordinance establishes minimum setbacks for energy facilities.
All proposed facilities must identify nearby roads and occupied buildings
before submitting their permit applications to the planning department.</p>
<p>Facilities must remain at least 125 feet from public roads and 250 feet
from occupied buildings. Applications must include a site plan showing each
structure, the property boundaries, and all applicable setback distances.</p>
<p>The planning department reviews these measurements before issuing a
permit. Operators must maintain the approved setbacks throughout the life
of the facility and report any changes to the approved site plan.</p>
<p id="result">Waiting for JavaScript</p>
<script>
document.getElementById('result').textContent = 'COMPASS ready';
</script>
</body></html>
"""
_TABLE_VALUES = ("roads", "125", "buildings", "250")


def _require(condition, message):
    """Fail even when Python assertions are disabled"""
    if not condition:
        raise COMPASSRuntimeError(message)


def _check_isolation():
    """Require an unprivileged process in a loopback-only namespace"""
    _require(os.geteuid() != 0, "Tests must not run as root")
    interfaces = {name for _, name in socket.if_nameindex()}
    _require(
        interfaces == {"lo"}, f"External interfaces present: {interfaces}"
    )


def _check_runtime_cache():
    """Require writable caches and pre-fetched Camoufox assets"""
    from camoufox.pkgman import INSTALL_DIR

    cache = Path(os.environ["XDG_CACHE_HOME"])
    _require(cache.is_absolute(), "XDG_CACHE_HOME must be absolute")
    with NamedTemporaryFile(dir=cache) as temporary:
        temporary.write(b"COMPASS cache write check")
        temporary.flush()
    _require(
        INSTALL_DIR.resolve() == Path("/opt/assets/cache/camoufox"),
        "Camoufox must use the pre-fetched browser assets",
    )
    _require(
        (INSTALL_DIR / "version.json").is_file(),
        "Pre-fetched Camoufox version information missing",
    )


def _check_page(browser, url):
    """Require navigation, DOM rendering, and JavaScript execution"""
    page = browser.new_page()
    try:
        response = page.goto(url, timeout=30000)
        _require(response.status == HTTPStatus.OK, "Browser navigation failed")
        page.wait_for_function(
            "document.querySelector('#result').textContent==='COMPASS ready'",
            timeout=30000,
        )
        _require(page.title() == "COMPASS checks", "Unexpected browser page")
    finally:
        page.close()


def _check_browser(module, url):
    """Launch a browser through its own installed driver"""
    with import_module(f"{module}.sync_api").sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, timeout=30000)
        try:
            _check_page(browser, url)
        finally:
            browser.close()


def _check_camoufox(url):
    """Launch Camoufox using its shared offline browser assets"""
    from camoufox.sync_api import Camoufox

    with Camoufox(headless=True, geoip=False) as browser:
        _check_page(browser, url)


def _check_scrapling(url):
    """Require HTTP and rendered browser responses from Scrapling"""
    from scrapling.fetchers import Fetcher, PlayWrightFetcher, StealthyFetcher

    response = Fetcher.get(url, timeout=30, retries=0)
    _require(response.status == HTTPStatus.OK, "Scrapling HTTP fetch failed")
    _require(
        response.css("h1::text").get() == "COMPASS fixture",
        "Scrapling HTTP content missing",
    )
    for fetcher in (PlayWrightFetcher, StealthyFetcher):
        print(f"Scrapling: {fetcher.__name__}", flush=True)
        response = fetcher.fetch(
            url, headless=True, google_search=False, timeout=30000
        )
        _require(
            response.status == HTTPStatus.OK, "Scrapling browser fetch failed"
        )
        _require(
            response.css("#result::text").get() == "COMPASS ready",
            "Scrapling did not render JavaScript",
        )


async def _check_crawl4ai(url, directory):
    """Require a successful crawl rather than setup or doctor output"""
    from crawl4ai import (
        AsyncWebCrawler,
        BrowserConfig,
        CacheMode,
        CrawlerRunConfig,
    )

    async with AsyncWebCrawler(
        config=BrowserConfig(headless=True, verbose=False),
        base_directory=str(directory),
    ) as crawler:
        result = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                page_timeout=30000,
                wait_for="css:#result",
            ),
        )
    _require(result.success, f"Crawl4AI failed: {result.error_message}")
    _require(result.status_code == HTTPStatus.OK, "Crawl4AI HTTP fetch failed")
    _require(
        "COMPASS ready" in result.markdown.raw_markdown,
        "Crawl4AI rendered content missing",
    )


def _write_scanned_pdf(path):
    """Create a raster-only PDF with known text and a ruled table"""
    from PIL import Image, ImageDraw, ImageFont

    with Image.new("RGB", (1700, 2200), "white") as image:
        drawing = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=48)
        drawing.text(
            (120, 150), "COMPASS COUNTY ORDINANCE", fill="black", font=font
        )
        rows = (
            ("Structure", "Setback"),
            ("Roads", "125 feet"),
            ("Buildings", "250 feet"),
        )
        for row_index, row in enumerate(rows):
            for column_index, text in enumerate(row):
                drawing.text(
                    (160 + column_index * 750, 450 + row_index * 180),
                    text,
                    fill="black",
                    font=font,
                )
        for horizontal in (400, 580, 760, 940):
            drawing.line((120, horizontal, 1580, horizontal), "black", 4)
        for vertical in (120, 870, 1580):
            drawing.line((vertical, 400, vertical, 940), "black", 4)
        image.save(path, format="PDF", resolution=200)


def _check_docling_result(result):
    """Reject empty OCR, partial conversion, and missing table cells"""
    _require(result.status.value == "success", "Docling conversion failed")
    markdown = result.document.export_to_markdown().casefold()
    _require("compass county" in markdown, "Docling OCR heading missing")
    _require(
        all(value in markdown for value in _TABLE_VALUES),
        "Docling OCR values missing",
    )
    for table in result.document.tables:
        cells = " ".join(
            cell.text for cell in table.data.table_cells
        ).casefold()
        if all(value in cells for value in _TABLE_VALUES):
            return
    _require(False, "Docling did not extract the scanned table cells")


def _check_docling(directory):
    """Convert a scanned PDF using native OCR and table models"""
    from docling_rs import DocumentConverter
    from PyPDF2 import PdfReader

    source = directory / "scanned.pdf"
    _write_scanned_pdf(source)
    _require(
        not any(
            page.extract_text().strip() for page in PdfReader(source).pages
        ),
        "OCR fixture unexpectedly contains embedded text",
    )
    result = DocumentConverter(
        do_ocr=True,
        force_full_page_ocr=True,
        do_table_structure=True,
        document_timeout=180,
    ).convert(source)
    _check_docling_result(result)


def _main():
    """Run all required checks against fresh local fixtures"""
    _check_isolation()
    home = Path.home()
    home.mkdir(parents=True, exist_ok=True)
    _check_runtime_cache()
    os.environ["CRAWL4_AI_BASE_DIRECTORY"] = str(home)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with TemporaryDirectory(prefix="compass-checks-") as temporary:
        directory = Path(temporary)
        (directory / "index.html").write_text(_HTML, encoding="utf-8")
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            partial(SimpleHTTPRequestHandler, directory=str(directory)),
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/index.html"
        try:
            for module in ("playwright", "rebrowser_playwright", "patchright"):
                print(f"Checking {module}", flush=True)
                _check_browser(module, url)
            print("Checking Camoufox", flush=True)
            _check_camoufox(url)
            print("Checking Scrapling", flush=True)
            _check_scrapling(url)
            print("Checking Crawl4AI", flush=True)
            asyncio.run(_check_crawl4ai(url, directory))
            print("Checking Docling OCR and tables", flush=True)
            _check_docling(directory)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    print("All offline non-root tests passed", flush=True)


if __name__ == "__main__":
    _main()
