"""COMPASS date extraction tests"""

from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

import pytest

from compass.extraction import date as date_module
from compass.extraction.date import (
    DateExtractor,
    _parse_date,
    _trim_pages,
    _MAX_HEAD_PAGES,
    _MAX_TAIL_PAGES,
)


def test_parse_date_empty_or_falsy():
    """Empty or falsy response yields all-None"""
    assert _parse_date(None) == (None, None, None)
    assert _parse_date({}) == (None, None, None)


def test_parse_date_full_date():
    """A complete, in-range date is returned as-is"""
    assert _parse_date({"year": 2023, "month": 10, "day": 15}) == (
        2023,
        10,
        15,
    )


def test_parse_date_year_only():
    """Missing month/day come back as None"""
    assert _parse_date({"year": 2023}) == (2023, None, None)


def test_parse_date_null_year_abstains():
    """A null year (model abstention) yields all-None"""
    assert _parse_date({"year": None, "month": 7, "day": 15}) == (
        None,
        7,
        15,
    )


def test_parse_date_out_of_range_elements_dropped():
    """Out-of-range elements are dropped to None; valid ones kept"""
    # month out of range
    assert _parse_date({"year": 2023, "month": 20, "day": 15}) == (
        2023,
        None,
        15,
    )
    # year before the floor
    assert _parse_date({"year": 1900, "month": 7, "day": 15}) == (
        None,
        7,
        15,
    )
    # year too far in the future (ceiling is next year)
    too_future = datetime.now().year + 2
    assert _parse_date({"year": too_future, "month": 7, "day": 15}) == (
        None,
        7,
        15,
    )
    # day out of range
    assert _parse_date({"year": 2023, "month": 7, "day": 32}) == (
        2023,
        7,
        None,
    )


def test_parse_date_coerces_strings_and_floats():
    """Numeric strings and floats are coerced to ints"""
    assert _parse_date({"year": "2023"}) == (2023, None, None)
    assert _parse_date({"year": 2020.0}) == (2020, None, None)
    assert _parse_date({"year": "2020.0"}) == (2020, None, None)


def test_parse_date_non_numeric_is_dropped():
    """A non-numeric value is dropped rather than raising"""
    assert _parse_date({"year": "n/a", "month": 7, "day": 15}) == (
        None,
        7,
        15,
    )


def test_trim_pages_short_doc_unchanged():
    """A short document (fewer pages than the cap) is returned as-is"""
    pages = ["p0", "p1", "p2", "p3", "p4"]
    assert _trim_pages(pages) == pages


def test_trim_pages_at_cap_unchanged():
    """A document exactly at the combined cap is returned unchanged"""
    pages = [str(i) for i in range(_MAX_HEAD_PAGES + _MAX_TAIL_PAGES)]
    assert _trim_pages(pages) == pages


def test_trim_pages_long_keeps_head_and_tail():
    """Long documents keep the first head pages and last tail pages"""
    pages = [str(i) for i in range(100)]
    trimmed = _trim_pages(pages)
    assert len(trimmed) == _MAX_HEAD_PAGES + _MAX_TAIL_PAGES
    assert trimmed[:_MAX_HEAD_PAGES] == pages[:_MAX_HEAD_PAGES]
    assert trimmed[_MAX_HEAD_PAGES:] == pages[-_MAX_TAIL_PAGES:]


def test_trim_pages_never_duplicates():
    """No page is duplicated across the head/tail at any document length"""
    for n in range(_MAX_HEAD_PAGES + _MAX_TAIL_PAGES + 5):
        pages = [str(i) for i in range(n)]
        trimmed = _trim_pages(pages)
        assert len(trimmed) == len(set(trimmed))


class _FakeCaller:
    """Records calls and returns a canned response dict"""

    def __init__(self, response):
        self._response = response
        self.calls = []

    async def call(self, sys_msg, content, **_kwargs):
        self.calls.append({"sys_msg": sys_msg, "content": content})
        return self._response


def _doc(source=None):
    return SimpleNamespace(attrs={"source": source} if source else {})


def _stub_pages(pages):
    """Return a ``raw_pages_from_doc`` replacement that yields ``pages``"""

    def _raw_pages(*_args, **_kwargs):
        return pages

    return _raw_pages


async def test_parse_reads_body_and_returns_date(monkeypatch):
    """parse() sends the document text and returns the parsed date"""
    monkeypatch.setattr(
        date_module, "raw_pages_from_doc", _stub_pages(["page text"])
    )
    caller = _FakeCaller({"year": 2021, "month": 6, "day": 1})
    extractor = DateExtractor(caller)

    assert await extractor.parse(_doc()) == (2021, 6, 1)
    assert len(caller.calls) == 1
    assert "page text" in caller.calls[0]["content"]


async def test_parse_includes_url_hint(monkeypatch):
    """The source URL is passed into the prompt as a hint"""
    monkeypatch.setattr(
        date_module, "raw_pages_from_doc", _stub_pages(["body"])
    )
    caller = _FakeCaller({"year": 2021})
    extractor = DateExtractor(caller)

    await extractor.parse(_doc(source="https://example.com/ord-2021.pdf"))
    assert "https://example.com/ord-2021.pdf" in caller.calls[0]["content"]


async def test_parse_no_text_skips_llm(monkeypatch):
    """With no document text, parse() returns all-None without calling LLM"""
    monkeypatch.setattr(date_module, "raw_pages_from_doc", _stub_pages([]))
    caller = _FakeCaller({"year": 2021})
    extractor = DateExtractor(caller)

    assert await extractor.parse(_doc()) == (None, None, None)
    assert caller.calls == []


async def test_parse_abstains_on_empty_response(monkeypatch):
    """An empty caller response yields all-None"""
    monkeypatch.setattr(
        date_module, "raw_pages_from_doc", _stub_pages(["body"])
    )
    extractor = DateExtractor(_FakeCaller({}))
    assert await extractor.parse(_doc()) == (None, None, None)


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
