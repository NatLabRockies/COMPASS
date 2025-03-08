"""Test COMPASS Ordinance location validation tests."""

import os
from pathlib import Path
from functools import partial

import pytest
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter

from elm import ApiBase
from elm.web.document import PDFDocument, HTMLDocument
from elm.utilities.parse import read_pdf
from compass.llm import StructuredLLMCaller
from compass.services.openai import OpenAIService
from compass.services.provider import RunningAsyncServices
from compass.utilities import RTS_SEPARATORS
from compass.validation.location import (
    CountyValidator,
    CountyNameValidator,
    CountyJurisdictionValidator,
    URLValidator,
    _validator_check_for_doc,
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
def oai_async_azure_client():
    """OpenAi Azure client to use for tests"""
    return openai.AsyncAzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_VERSION"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    )


@pytest.fixture()
def structured_llm_caller():
    """StructuredLLMCaller instance for testing"""
    return StructuredLLMCaller(
        llm_service=OpenAIService,
        model="gpt-4",
        temperature=0,
        seed=42,
        timeout=30,
    )


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


@pytest.mark.skipif(SHOULD_SKIP, reason="requires Azure OpenAI key")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "county,state,url,truth",
    [
        (
            "El Paso",
            "Indiana",
            "https://programs.dsireusa.org/system/program/detail/4332/"
            "madison-county-wind-energy-systems-ordinance",
            False,
        ),
        (
            "Madison",
            "Indiana",
            "https://programs.dsireusa.org/system/program/detail/4332/"
            "madison-county-wind-energy-systems-ordinance",
            False,
        ),
        (
            "Madison",
            "North Carolina",
            "https://programs.dsireusa.org/system/program/detail/4332/"
            "madison-county-wind-energy-systems-ordinance",
            False,
        ),
        (
            "Decatur",
            "Indiana",
            "http://www.decaturcounty.in.gov/doc/area-plan-commission/updates/"
            "zoning_ordinance_-_article_13_wind_energy_conversion_system_"
            "(WECS).pdf",
            True,
        ),
        (
            "Decatur",
            "Colorado",
            "http://www.decaturcounty.in.gov/doc/area-plan-commission/updates/"
            "zoning_ordinance_-_article_13_wind_energy_conversion_system_"
            "(WECS).pdf",
            False,
        ),
        (
            "El Paso",
            "Indiana",
            "http://www.decaturcounty.in.gov/doc/area-plan-commission/updates/"
            "zoning_ordinance_-_article_13_wind_energy_conversion_system_"
            "(WECS).pdf",
            False,
        ),
    ],
)
async def test_url_matches_county(
    oai_async_azure_client, structured_llm_caller, county, state, url, truth
):
    """Test the URL validator class (basic execution)"""
    url_validator = URLValidator(structured_llm_caller)
    services = [OpenAIService(oai_async_azure_client, rate_limit=50_000)]
    async with RunningAsyncServices(services):
        out = await url_validator.check(url, county=county, state=state)
        assert out == truth


@pytest.mark.skipif(SHOULD_SKIP, reason="requires Azure OpenAI key")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "county,doc_fn,truth",
    [
        ("Decatur", "indiana_general_ord.pdf", False),
        ("Decatur", "Decatur Indiana.pdf", True),
        ("Hamlin", "Hamlin South Dakota.pdf", True),
        ("Atlantic", "Atlantic New Jersey.txt", False),
        ("Barber", "Barber Kansas.pdf", False),
    ],
)
async def test_doc_matches_county_jurisdiction(
    oai_async_azure_client,
    structured_llm_caller,
    county,
    doc_fn,
    truth,
    test_data_dir,
):
    """Test the `CountyJurisdictionValidator` class (basic execution)"""
    doc = _load_doc(test_data_dir, doc_fn)
    cj_validator = CountyJurisdictionValidator(structured_llm_caller)
    services = [OpenAIService(oai_async_azure_client, rate_limit=100_000)]
    async with RunningAsyncServices(services):
        out = await _validator_check_for_doc(
            doc=doc, validator=cj_validator, county=county
        )
        assert out == truth


@pytest.mark.skipif(SHOULD_SKIP, reason="requires Azure OpenAI key")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "county,state,doc_fn,truth",
    [
        ("Decatur", "Indiana", "Decatur Indiana.pdf", True),
        ("Hamlin", "South Dakota", "Hamlin South Dakota.pdf", True),
        ("Anoka", "Minnesota", "Anoka Minnesota.txt", True),
    ],
)
async def test_doc_matches_county_name(
    oai_async_azure_client,
    structured_llm_caller,
    county,
    state,
    doc_fn,
    truth,
    test_data_dir,
):
    """Test the `CountyNameValidator` class (basic execution)"""
    doc = _load_doc(test_data_dir, doc_fn)
    cn_validator = CountyNameValidator(structured_llm_caller)
    services = [OpenAIService(oai_async_azure_client, rate_limit=100_000)]
    async with RunningAsyncServices(services):
        out = await _validator_check_for_doc(
            doc=doc, validator=cn_validator, county=county, state=state
        )
        assert out == truth


@pytest.mark.skipif(SHOULD_SKIP, reason="requires Azure OpenAI key")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "county,state,doc_fn,url,truth",
    [
        (
            "Decatur",
            "Indiana",
            "Decatur Indiana.pdf",
            "http://www.decaturcounty.in.gov/doc/area-plan-commission/z.pdf",
            True,
        ),
        (
            "Hamlin",
            "South Dakota",
            "Hamlin South Dakota.pdf",
            "http://www.test.gov",
            True,
        ),
        (
            "Anoka",
            "Minnesota",
            "Anoka Minnesota.txt",
            "http://www.test.gov",
            False,
        ),
        (
            "Atlantic",
            "New Jersey",
            "Atlantic New Jersey.txt",
            "http://www.test.gov",
            False,
        ),
    ],
)
async def test_doc_matches_county(
    oai_async_azure_client,
    structured_llm_caller,
    county,
    state,
    doc_fn,
    url,
    truth,
    test_data_dir,
):
    """Test the `CountyValidator` class (basic execution)"""
    doc = _load_doc(test_data_dir, doc_fn)
    doc.attrs["source"] = url

    county_validator = CountyValidator(structured_llm_caller)
    services = [OpenAIService(oai_async_azure_client, rate_limit=100_000)]
    async with RunningAsyncServices(services):
        out = await county_validator.check(doc=doc, county=county, state=state)
        assert out == truth


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
