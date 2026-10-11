"""COMPASS Ordinance jurisdiction integration tests"""

import asyncio
from pathlib import Path

import pytest

from compass.utilities.jurisdictions import (
    Jurisdiction,
    jurisdictions_from_df,
    load_all_jurisdiction_info,
    jurisdiction_websites,
)
from compass.validation.location import DTreeURLJurisdictionValidator


def test_load_all_jurisdictions():
    """Test the `load_all_jurisdiction_info` function"""

    jurisdiction_info = load_all_jurisdiction_info()
    assert not jurisdiction_info.empty

    expected_cols = [
        "County",
        "State",
        "Subdivision",
        "Jurisdiction Type",
        "FIPS",
        "Website",
    ]
    assert all(col in jurisdiction_info for col in expected_cols)
    for g, data in jurisdiction_info.groupby(
        ["County", "State", "Subdivision", "Jurisdiction Type"]
    ):
        if len(data) > 1:
            print(g)
            print(data)
    assert len(jurisdiction_info) == len(
        jurisdiction_info.groupby(
            ["County", "State", "Subdivision", "Jurisdiction Type"]
        )
    )
    assert len(jurisdiction_info) == len(jurisdiction_info.groupby(["FIPS"]))

    # Spot checks:
    assert "Decatur" in set(jurisdiction_info["County"])
    assert "Box Elder" in set(jurisdiction_info["County"])
    assert "Colorado" in set(jurisdiction_info["State"])
    assert "Rhode Island" in set(jurisdiction_info["State"])


def test_load_all_jurisdictions_returns_shallow_copy():
    """Test cached jurisdiction info is returned as a caller-safe copy"""

    jurisdiction_info = load_all_jurisdiction_info()
    county_col = jurisdiction_info.columns.get_loc("County")
    original_county = jurisdiction_info.iloc[0, county_col]

    jurisdiction_info.iloc[0, county_col] = "Modified County"

    fresh_jurisdiction_info = load_all_jurisdiction_info()

    assert fresh_jurisdiction_info is not jurisdiction_info
    assert fresh_jurisdiction_info.iloc[0, county_col] == original_county


def test_jurisdiction_websites():
    """Test the `jurisdiction_websites` function"""

    websites = jurisdiction_websites()
    assert len(websites) == len(load_all_jurisdiction_info())
    assert isinstance(websites, dict)

    # Spot checks:
    assert "18031" in websites  # Decatur Indiana
    assert "08041" in websites  # El Paso, Colorado
    assert "49003" in websites  # Box Elder, Utah


@pytest.mark.asyncio
async def test_url_matches_known_jurisdiction_website_skips_llm(monkeypatch):
    """Test URL validation passes when canonical website domain matches"""
    jurisdiction_info = load_all_jurisdiction_info()
    jurisdiction = next(
        jur
        for jur in jurisdictions_from_df(jurisdiction_info)
        if jur.website_url
    )
    website_url = jurisdiction.website_url
    jurisdiction = Jurisdiction(
        jurisdiction.type,
        state=jurisdiction.state,
        county=jurisdiction.county,
        subdivision_name=jurisdiction.subdivision_name,
        code=jurisdiction.code,
    )
    url = f"{website_url.rstrip('/')}/ordinances/test.pdf"

    async def _should_not_run(*args, **kwargs):
        await asyncio.sleep(0)
        raise AssertionError("LLM validation should have been skipped")

    monkeypatch.setattr(
        "compass.validation.location.run_async_tree",
        _should_not_run,
    )

    url_validator = DTreeURLJurisdictionValidator(
        jurisdiction, llm_service=object()
    )
    assert await url_validator.check(url)


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
