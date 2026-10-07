"""Test COMPASS Ordinance logging logic"""

from pathlib import Path

import pytest

from compass.pipeline.data_classes import (
    WebSearchParams, ProcessRequest, CollectionRequest,
)
from compass.pipeline.runtime import PipelineRuntime


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


@pytest.mark.parametrize("mode", ["process", "collect"])
@pytest.mark.parametrize("priority_search", [{}, {"max_pages": 7}])
def test_priority_settings_reach_runtime(mode, priority_search):
    """Both supported commands pass queue limits through their requests."""
    cls = ProcessRequest if mode == "process" else CollectionRequest
    request = cls(
        out_dir="unused",
        tech="data_centers",
        jurisdiction_fp="unused.csv",
        model="gpt-4o-mini",
        priority_search=priority_search,
    )
    runtime = PipelineRuntime(request)
    assert runtime.search_params.priority_search == priority_search
    assert runtime._llm_services


@pytest.mark.parametrize("mode", ["process", "collect"])
@pytest.mark.parametrize("settings", [{}, {"priority_search": None}])
def test_standard_search_is_default(mode, settings):
    """Omitted and null settings keep priority collection disabled."""
    cls = ProcessRequest if mode == "process" else CollectionRequest
    request = cls(
        out_dir="unused",
        tech="data_centers",
        jurisdiction_fp="unused.csv",
        model="gpt-4o-mini",
        **settings,
    )
    runtime = PipelineRuntime(request)
    assert runtime.search_params.priority_search is None
    if mode == "collect":
        assert runtime._llm_services == []
