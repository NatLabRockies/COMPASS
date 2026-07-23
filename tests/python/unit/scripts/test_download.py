"""Tests for compass.scripts.download"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import compass.scripts.download as download_module
from compass.utilities.enums import LLMTasks


@pytest.mark.asyncio
async def test_find_jurisdiction_website_returns_base_domain(monkeypatch):
    """Return the canonical root URL for the selected website"""

    async def fake_search_with_fallback(**_kwargs):  # ruff:ignore[unused-async]
        return [
            "https://prattvilleal.gov/venue/autauga-county-commission/",
            "https://prattvilleal.gov/government/mayor",
            "https://example.org/other-page",
        ]

    class DummyValidator:
        def __init__(self, **_kwargs):
            pass

        async def check(self, url, jurisdiction):
            return url == "https://prattvilleal.gov/"

    monkeypatch.setattr(
        download_module,
        "search_with_fallback",
        fake_search_with_fallback,
    )
    monkeypatch.setattr(
        download_module,
        "JurisdictionWebsiteValidator",
        DummyValidator,
    )

    jurisdiction = SimpleNamespace(
        full_name="Autauga County, Alabama",
        full_name_the_prefixed="Autauga County, Alabama",
    )
    model_config = SimpleNamespace(
        llm_service=object(),
        llm_call_kwargs={},
    )

    out = await download_module.find_jurisdiction_website(
        jurisdiction, {LLMTasks.DEFAULT: model_config}
    )

    assert out == "https://prattvilleal.gov/"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
