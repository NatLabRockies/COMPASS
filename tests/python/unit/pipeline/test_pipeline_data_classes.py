"""Test COMPASS Ordinance logging logic"""

from pathlib import Path

import pytest

from compass.pipeline import ProcessRequest
from compass.pipeline.data_classes import WebSearchParams


def test_wsp_se_kwargs():
    """Test the `se_kwargs` property of `WebSearchParams`"""

    assert not WebSearchParams().se_kwargs

    expected = {
        "pw_google_se_kwargs": {},
        "search_engines": ["PlaywrightGoogleLinkSearch"],
    }
    assert (
        WebSearchParams(
            search_engines=[{"se_name": "PlaywrightGoogleLinkSearch"}]
        ).se_kwargs
        == expected
    )

    expected = {
        "pw_google_se_kwargs": {"use_homepage": False},
        "search_engines": ["PlaywrightGoogleLinkSearch"],
    }
    assert (
        WebSearchParams(
            search_engines=[
                {
                    "se_name": "PlaywrightGoogleLinkSearch",
                    "use_homepage": False,
                }
            ]
        ).se_kwargs
        == expected
    )

    expected = {
        "ddg_api_kwargs": {"timeout": 300, "backend": "html", "verify": False},
        "pw_google_se_kwargs": {"use_homepage": False},
        "search_engines": [
            "PlaywrightGoogleLinkSearch",
            "APIDuckDuckGoSearch",
        ],
    }
    assert (
        WebSearchParams(
            search_engines=[
                {
                    "se_name": "PlaywrightGoogleLinkSearch",
                    "use_homepage": False,
                },
                {
                    "se_name": "APIDuckDuckGoSearch",
                    "timeout": 300,
                    "backend": "html",
                    "verify": False,
                },
            ]
        ).se_kwargs
        == expected
    )


def test_request_models_accepts_runtime_rate_tracker(tmp_path):
    """Build model configs without passing runtime state to them"""
    request = ProcessRequest(
        out_dir=tmp_path / "outputs",
        tech="solar",
        jurisdiction_fp=tmp_path / "jurisdictions.csv",
        model=[{"name": "gpt-4o-mini", "client_type": "openai"}],
    )

    models = request.models

    assert models["default"].name == "gpt-4o-mini"
    assert request.rate_tracker is not None


def test_wsp_url_filter_defaults_are_isolated():
    """Custom URL filters should not leak into later requests"""
    custom = WebSearchParams(
        url_ignore_substrings=["blocked.example"],
        url_keep_substrings=["trusted.example"],
    )
    defaults = WebSearchParams()

    assert "blocked.example" in custom.url_ignore_substrings
    assert "trusted.example" in custom.url_keep_substrings
    assert "blocked.example" not in defaults.url_ignore_substrings
    assert "trusted.example" not in defaults.url_keep_substrings


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
