"""Tests for page-level jurisdiction events in event logs"""

import json

import pytest
from elm.web.document import PDFDocument

from compass.utilities.event_log import EventLog, emit_event, use_event_log
from compass.validation.location import (
    DTreeJurisdictionValidator,
    DTreeURLJurisdictionValidator,
    JurisdictionValidator,
    JurisdictionWebsiteValidator,
    _validator_check_for_doc,
)


class PageValidator:
    """Validator stand-in that accepts pages mentioning the town"""

    META_SCORE_KEY = "Jurisdiction Validation Score"

    async def check(self, text):
        emit_event("page_checked")
        return "Marana" in text


@pytest.mark.asyncio
async def test_pages_are_tagged_and_the_vote_is_recorded(tmp_path):
    """Each page's checks carry its page number; the vote is logged"""
    doc = PDFDocument(["Town of Marana ordinance", "Building exteriors"])
    doc.attrs["source"] = "https://maranaaz.gov/ord.pdf"
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        passed = await _validator_check_for_doc(
            PageValidator(), doc, score_thresh=0.6
        )
    events = [json.loads(x) for x in log.path.read_text().splitlines()]
    checked = [e for e in events if e["type"] == "page_checked"]
    assert sorted(e["tags"]["page"] for e in checked) == [1, 2]
    assert {e["tags"]["pages"] for e in checked} == {2}
    (vote,) = [e for e in events if e["type"] == "jurisdiction_vote"]
    assert [p["verdict"] for p in vote["pages"]] == [True, False]
    assert vote["threshold"] == pytest.approx(0.6)
    assert vote["passed"] is passed
    assert vote["score"] == pytest.approx(
        len("Town of Marana ordinance")
        / (len("Town of Marana ordinance") + len("Building exteriors"))
    )


@pytest.mark.asyncio
async def test_website_domain_shortcut_is_recorded(tmp_path, monkeypatch):
    """A URL on the jurisdiction's own site passes without LLM calls"""
    monkeypatch.setattr(
        "compass.validation.location._url_matches_known_jurisdiction_website",
        lambda *_args: True,
    )
    validator = DTreeURLJurisdictionValidator(
        jurisdiction="Albemarle County, Virginia", llm_service=None
    )
    log = EventLog(tmp_path / "events.jsonl")
    url = "https://lfweb.albemarle.org/WebLink/DocView.aspx?id=1"
    with use_event_log(log):
        assert await validator.check(url)
    (event,) = [json.loads(x) for x in log.path.read_text().splitlines()]
    assert event["type"] == "jurisdiction_url_check"
    assert event["passed"] is True
    assert event["method"] == "domain matches the jurisdiction's website"


def _branches(events):
    """(step, outcome) of each branch event, in order"""
    return [
        (e["tags"]["step"], e["outcome"])
        for e in events
        if e["type"] == "branch"
    ]


def _read(log):
    return [json.loads(x) for x in log.path.read_text().splitlines()]


@pytest.mark.asyncio
async def test_page_vote_records_its_steps_and_branch(tmp_path):
    """Page checks and the vote link to their workflow nodes"""
    doc = PDFDocument(["Town of Marana ordinance", "Building exteriors"])
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        await _validator_check_for_doc(PageValidator(), doc, score_thresh=0.9)
    events = _read(log)
    checked = [e for e in events if e["type"] == "page_checked"]
    assert {e["tags"]["step"] for e in checked} == {
        "filter.jurisdiction.pages.tree"
    }
    (vote,) = [e for e in events if e["type"] == "jurisdiction_vote"]
    assert vote["tags"]["step"] == "filter.jurisdiction.vote"
    assert _branches(events) == [("filter.jurisdiction.vote", "fail")]
    (branch,) = [e for e in events if e["type"] == "branch"]
    assert branch["threshold"] == pytest.approx(0.9)
    assert branch["score"] == pytest.approx(vote["score"])


@pytest.mark.asyncio
async def test_jurisdiction_check_records_each_decision(tmp_path, monkeypatch):
    """A URL that fails both URL checks goes to a page vote"""
    monkeypatch.setattr(
        "compass.validation.location._url_matches_known_jurisdiction_website",
        lambda *_args: False,
    )

    async def url_tree(_self, url):  # ruff:ignore[unused-async]
        emit_event("url_tree_ran", url=url)
        return False

    async def page_check(_self, text):  # ruff:ignore[unused-async]
        return "Marana" in text

    monkeypatch.setattr(
        DTreeURLJurisdictionValidator, "check_with_tree", url_tree
    )
    monkeypatch.setattr(DTreeJurisdictionValidator, "check", page_check)
    doc = PDFDocument(["Town of Marana ordinance"])
    doc.attrs["source"] = "https://example.com/ord.pdf"
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        passed = await JurisdictionValidator(llm_service=None).check(
            doc, jurisdiction="Town of Marana, Arizona"
        )
    events = _read(log)
    assert passed
    assert _branches(events) == [
        ("filter.jurisdiction.domain", "no"),
        ("filter.jurisdiction.url_tree", "fail"),
        ("filter.jurisdiction.vote", "pass"),
    ]
    (tree,) = [e for e in events if e["type"] == "url_tree_ran"]
    assert tree["tags"]["step"] == "filter.jurisdiction.url_tree"


@pytest.mark.asyncio
async def test_known_website_passes_on_the_domain_alone(tmp_path, monkeypatch):
    """A URL on the jurisdiction's site records only the domain check"""
    monkeypatch.setattr(
        "compass.validation.location._url_matches_known_jurisdiction_website",
        lambda *_args: True,
    )
    doc = PDFDocument(["Anything"])
    doc.attrs["source"] = "https://maranaaz.gov/ord.pdf"
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        assert await JurisdictionValidator(llm_service=None).check(
            doc, jurisdiction="Town of Marana, Arizona"
        )
    events = _read(log)
    assert _branches(events) == [("filter.jurisdiction.domain", "yes")]
    (check,) = [e for e in events if e["type"] == "jurisdiction_url_check"]
    assert check["tags"]["step"] == "filter.jurisdiction.domain"


@pytest.mark.asyncio
async def test_document_without_url_goes_straight_to_pages(
    tmp_path, monkeypatch
):
    """No URL to check: the page vote decides"""

    async def page_check(_self, text):  # ruff:ignore[unused-async]
        return False

    monkeypatch.setattr(DTreeJurisdictionValidator, "check", page_check)
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        assert not await JurisdictionValidator(llm_service=None).check(
            PDFDocument(["Some page"]), jurisdiction="Town of Marana, Arizona"
        )
    assert _branches(_read(log)) == [
        ("filter.jurisdiction.domain", "no URL"),
        ("filter.jurisdiction.vote", "fail"),
    ]


@pytest.mark.asyncio
async def test_website_candidate_checks_are_recorded(tmp_path, monkeypatch):
    """A candidate failing the URL check is judged by its page"""

    async def url_check(_self, url):  # ruff:ignore[unused-async]
        return False

    async def page_is_main(_self, url, jurisdiction):  # ruff:ignore[unused-async]
        return True

    monkeypatch.setattr(DTreeURLJurisdictionValidator, "check", url_check)
    monkeypatch.setattr(
        JurisdictionWebsiteValidator, "_page_is_main_website", page_is_main
    )
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        assert await JurisdictionWebsiteValidator(llm_service=None).check(
            "https://maranaaz.gov", "Town of Marana, Arizona"
        )
    assert _branches(_read(log)) == [
        ("collect.website.url_check", "fail"),
        ("collect.website.page_check", "yes"),
    ]
