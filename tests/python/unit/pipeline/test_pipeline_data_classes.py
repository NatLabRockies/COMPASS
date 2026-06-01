"""Test COMPASS Ordinance logging logic"""

from pathlib import Path

import pytest

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


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
