"""Test COMPASS Ordinance logging logic"""

from pathlib import Path

import pytest

from compass.utilities.base import title_preserving_caps, WebSearchParams


def test_title_preserving_caps():
    """Test the `title_preserving_caps` function"""

    assert title_preserving_caps("hello world") == "Hello World"
    assert title_preserving_caps("hello World") == "Hello World"
    assert title_preserving_caps("Hello world") == "Hello World"
    assert title_preserving_caps("Hello World") == "Hello World"
    assert title_preserving_caps("HELLO WORLD") == "HELLO WORLD"

    assert title_preserving_caps("St. mcLean") == "St. McLean"


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


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
