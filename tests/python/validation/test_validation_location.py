"""Test COMPASS Ordinance location validation tests"""

import asyncio
import os
from pathlib import Path
from functools import partial

import pytest
import openai
from flaky import flaky
from langchain.text_splitter import RecursiveCharacterTextSplitter

from elm import ApiBase
from elm.web.document import PDFDocument, HTMLDocument
from elm.utilities.parse import read_pdf
from compass.services.openai import OpenAIService
from compass.services.provider import RunningAsyncServices
from compass.utilities import RTS_SEPARATORS
from compass.utilities.location import Jurisdiction
from compass.validation.location import (
    JurisdictionValidator,
    DTreeJurisdictionValidator,
    DTreeURLCountyValidator,
    _validator_check_for_doc,
    _weighted_vote,
)


SHOULD_SKIP = os.getenv("AZURE_OPENAI_API_KEY") is None
TESTING_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    RTS_SEPARATORS,
    chunk_size=3000,
    chunk_overlap=300,
    length_function=partial(ApiBase.count_tokens, model="gpt-4"),
    is_separator_regex=True,
)


@pytest.fixture(scope="module")
def event_loop():
    """Override default event loop fixture to make it module-level"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def oai_async_azure_client():
    """OpenAi Azure client to use for tests"""
    return openai.AsyncAzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_VERSION"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    )


@pytest.fixture(scope="module")
def oai_llm_service(oai_async_azure_client):
    """OpenAi Azure client to use for tests"""
    model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini")
    return OpenAIService(
        client=oai_async_azure_client, model_name=model_name, rate_limit=1e6
    )


@pytest.fixture(scope="module", autouse=True)
async def running_openai_service(oai_llm_service):
    """Set up running OpenAI service to use for tests"""
    async with RunningAsyncServices([oai_llm_service]):
        yield


def _load_doc(test_data_dir, doc_fn):
    """Load PDF or HTML doc for tests"""
    doc_fp = test_data_dir / doc_fn
    if doc_fp.suffix == ".pdf":
        with doc_fp.open("rb") as fh:
            pages = read_pdf(fh.read())
            return PDFDocument(pages)

    with doc_fp.open("r", encoding="utf-8") as fh:
        text = fh.read()
        return HTMLDocument([text], text_splitter=TESTING_TEXT_SPLITTER)


@flaky(max_runs=3, min_passes=1)
@pytest.mark.skipif(SHOULD_SKIP, reason="requires Azure OpenAI key")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "loc,url,truth",
    [
        (
            Jurisdiction("county", state="Indiana", county="El Paso"),
            "https://programs.dsireusa.org/system/program/detail/4332/"
            "madison-county-wind-energy-systems-ordinance",
            False,
        ),
        (
            Jurisdiction("county", state="Indiana", county="Madison"),
            "https://programs.dsireusa.org/system/program/detail/4332/"
            "madison-county-wind-energy-systems-ordinance",
            False,
        ),
        (
            Jurisdiction("county", state="North Carolina", county="Madison"),
            "https://programs.dsireusa.org/system/program/detail/4332/"
            "madison-county-wind-energy-systems-ordinance",
            False,
        ),
        (
            Jurisdiction("county", state="Indiana", county="Decatur"),
            "http://www.decaturcounty.in.gov/doc/area-plan-commission/updates/"
            "zoning_ordinance_-_article_13_wind_energy_conversion_system_"
            "(WECS).pdf",
            True,
        ),
        (
            Jurisdiction("county", state="Colorado", county="Decatur"),
            "http://www.decaturcounty.in.gov/doc/area-plan-commission/updates/"
            "zoning_ordinance_-_article_13_wind_energy_conversion_system_"
            "(WECS).pdf",
            False,
        ),
        (
            Jurisdiction("county", state="Indiana", county="El Paso"),
            "http://www.decaturcounty.in.gov/doc/area-plan-commission/updates/"
            "zoning_ordinance_-_article_13_wind_energy_conversion_system_"
            "(WECS).pdf",
            False,
        ),
        (
            Jurisdiction(
                "town", state="New York", subdivision_name="Allegany"
            ),
            "https://www.allegany.ny.org/uploads/1/4/0/1/140198361/"
            "town_of_allegany_solar_energy_local_law_v2_rev_040122.pdf",
            True,
        ),
    ],
)
async def test_url_matches_county(oai_llm_service, loc, url, truth):
    """Test the DTreeURLCountyValidator class (basic execution)"""
    url_validator = DTreeURLCountyValidator(
        loc, llm_service=oai_llm_service, temperature=0, seed=42, timeout=30
    )
    out = await url_validator.check(url)
    assert out == truth


@flaky(max_runs=3, min_passes=1)
@pytest.mark.skipif(SHOULD_SKIP, reason="requires Azure OpenAI key")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "loc,doc_fn,truth",
    [
        (
            Jurisdiction("county", state="Indiana", county="Decatur"),
            "indiana_general_ord.pdf",
            False,
        ),
        (
            Jurisdiction("county", state="Indiana", county="Decatur"),
            "Decatur Indiana.pdf",
            True,
        ),
        (
            Jurisdiction("county", state="South Dakota", county="Hamlin"),
            "Hamlin South Dakota.pdf",
            True,
        ),
        (
            Jurisdiction("county", state="New Jersey", county="Atlantic"),
            "Atlantic New Jersey.txt",
            False,
        ),
        (
            Jurisdiction(
                "city",
                state="New Jersey",
                county="Atlantic",
                subdivision_name="Linwood",
            ),
            "Atlantic New Jersey.txt",
            True,
        ),
        (
            Jurisdiction("county", state="Kansas", county="Barber"),
            "Barber Kansas.pdf",
            False,
        ),
        (
            Jurisdiction("county", state="Minnesota", county="Anoka"),
            "Anoka Minnesota.txt",
            False,
        ),
        (
            Jurisdiction("county", state="New York", county="Allegany"),
            "Allegany New York.pdf",
            False,
        ),
        (
            Jurisdiction(
                "town", state="New York", subdivision_name="Allegany"
            ),
            "Allegany New York.pdf",
            True,
        ),
    ],
)
async def test_doc_text_matches_jurisdiction(
    oai_llm_service, loc, doc_fn, truth, test_data_dir
):
    """Test the `DTreeJurisdictionValidator` class"""
    doc = _load_doc(test_data_dir, doc_fn)
    cj_validator = DTreeJurisdictionValidator(
        loc, llm_service=oai_llm_service, temperature=0, seed=42, timeout=30
    )
    out = await _validator_check_for_doc(doc=doc, validator=cj_validator)
    assert out == truth


@flaky(max_runs=3, min_passes=1)
@pytest.mark.skipif(SHOULD_SKIP, reason="requires Azure OpenAI key")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "loc,doc_fn,url,truth",
    [
        (
            Jurisdiction("county", state="Indiana", county="Decatur"),
            "Decatur Indiana.pdf",
            "http://www.decaturcounty.in.gov/doc/area-plan-commission/z.pdf",
            True,
        ),
        (
            Jurisdiction("county", state="South Dakota", county="Hamlin"),
            "Hamlin South Dakota.pdf",
            "http://www.test.gov",
            True,
        ),
        (
            Jurisdiction("county", state="Minnesota", county="Anoka"),
            "Anoka Minnesota.txt",
            "http://www.test.gov",
            False,
        ),
        (
            Jurisdiction("county", state="New Jersey", county="Atlantic"),
            "Atlantic New Jersey.txt",
            "http://www.test.gov",
            False,
        ),
    ],
)
async def test_doc_matches_jurisdiction(
    oai_llm_service, loc, doc_fn, url, truth, test_data_dir
):
    """Test the `JurisdictionValidator` class (basic execution)"""
    doc = _load_doc(test_data_dir, doc_fn)
    doc.attrs["source"] = url

    county_validator = JurisdictionValidator(
        llm_service=oai_llm_service, temperature=0, seed=42, timeout=30
    )
    out = await county_validator.check(doc=doc, jurisdiction=loc)
    assert out == truth


@pytest.mark.parametrize(
    "test_case",
    (
        (["one", "two", "three"], [1, 1, 0], (1 * 3 + 1 * 3) / (3 + 3 + 5)),
        (["one", "two", "three"], [1, None, 0], (1 * 3) / (3 + 5)),
    ),
)
def test_weighted_vote(test_case):
    """Test that the _weighted_vote function computes score properly"""
    pages, verdict, expected_score = test_case
    assert _weighted_vote(verdict, PDFDocument(pages)) == expected_score


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
