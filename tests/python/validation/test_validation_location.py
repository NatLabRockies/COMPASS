"""Test COMPASS Ordinance location validation tests"""

import os
from pathlib import Path

import pytest
from flaky import flaky
from elm.web.document import PDFDocument
from elm.utilities.parse import read_pdf_ocr

from compass.utilities.location import Jurisdiction
from compass.validation.location import (
    JurisdictionValidator,
    DTreeJurisdictionValidator,
    DTreeURLJurisdictionValidator,
    _validator_check_for_doc,
    _weighted_vote,
)


SHOULD_SKIP = os.getenv("AZURE_OPENAI_API_KEY") is None
PYT_CMD = os.getenv("TESSERACT_CMD")


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
    """Test the DTreeURLJurisdictionValidator class (basic execution)"""
    url_validator = DTreeURLJurisdictionValidator(
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
    oai_llm_service, loc, doc_fn, truth, doc_loader
):
    """Test the `DTreeJurisdictionValidator` class"""
    doc = doc_loader(doc_fn)
    cj_validator = DTreeJurisdictionValidator(
        loc, llm_service=oai_llm_service, temperature=0, seed=42, timeout=30
    )
    out = await _validator_check_for_doc(doc=doc, validator=cj_validator)
    assert out == truth


@pytest.mark.skipif(
    SHOULD_SKIP or not PYT_CMD,
    reason="requires Azure OpenAI key *and* PyTesseract command to be set",
)
async def test_doc_text_matches_jurisdiction_ocr(
    oai_llm_service, test_data_dir
):
    """Test the `DTreeJurisdictionValidator` class for scanned doc"""
    import pytesseract  # noqa: PLC0415

    pytesseract.pytesseract.tesseract_cmd = PYT_CMD

    loc = Jurisdiction("county", state="Kansas", county="Sedgwick")

    doc_fp = test_data_dir / "Sedgwick Kansas.pdf"
    with doc_fp.open("rb") as fh:
        pages = read_pdf_ocr(fh.read())
        doc = PDFDocument(pages)

    cj_validator = DTreeJurisdictionValidator(
        loc, llm_service=oai_llm_service, temperature=0, seed=42, timeout=30
    )
    assert await _validator_check_for_doc(doc=doc, validator=cj_validator)


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
    oai_llm_service, loc, doc_fn, url, truth, doc_loader
):
    """Test the `JurisdictionValidator` class (basic execution)"""
    doc = doc_loader(doc_fn)
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
