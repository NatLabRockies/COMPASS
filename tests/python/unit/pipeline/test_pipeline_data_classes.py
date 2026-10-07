"""Test COMPASS Ordinance logging logic"""

from pathlib import Path

import pytest

from compass.pipeline import (
    BaseRequest,
    CollectionRequest,
    ExtractionRequest,
    ProcessRequest,
)
from compass.pipeline.data_classes import WebSearchParams
from compass.pipeline.runtime import PipelineRuntime
from compass.utilities.io import ConfigType, load_config


@pytest.mark.parametrize("config_type", list(ConfigType))
@pytest.mark.parametrize(
    "request_class, source_dir_key, extra_config",
    [
        (BaseRequest, "ordinance_file_dir", {}),
        (ProcessRequest, "ordinance_file_dir", {}),
        (CollectionRequest, "source_file_dir", {}),
        (
            ExtractionRequest,
            "ordinance_file_dir",
            {"collection_manifest_fp": "./manifest.json"},
        ),
    ],
)
def test_request_from_inherited_config(
    tmp_path, config_type, request_class, source_dir_key, extra_config
):
    """Create requests from inherited configs with child overrides"""
    parent_dir = tmp_path / "parents"
    parent_dir.mkdir()
    parent = parent_dir / "parent.json"
    child = tmp_path / f"child.{config_type}"
    ConfigType.JSON.write(
        parent,
        {
            "out_dir": "./parent_outputs",
            "tech": "solar",
            "jurisdiction_fp": "./jurisdictions.csv",
            source_dir_key: "./sources",
            "model": "gpt-4o-mini",
            "max_num_concurrent_jurisdictions": 3,
            "log_level": "WARNING",
            "file_loader_kwargs": {
                "pw_launch_kwargs": {"headless": True, "timeout": 1000}
            },
        },
    )
    config_type.write(
        child,
        {
            "inherit_from": "parents/parent.json",
            "out_dir": "./child_outputs",
            "log_level": "DEBUG",
            "file_loader_kwargs": {"pw_launch_kwargs": {"timeout": 2000}},
            **extra_config,
        },
    )

    config = load_config(child)
    request = request_class(**config)

    assert "inherit_from" not in config
    assert request.tech == "solar"
    assert (
        request.jurisdiction_fp
        == (parent_dir / "jurisdictions.csv").as_posix()
    )
    assert (
        request.output_settings.out_dir
        == (tmp_path / "child_outputs").as_posix()
    )
    assert (
        request.output_settings.ordinance_file_dir
        == (parent_dir / "sources").as_posix()
    )
    assert request.user_model_input == "gpt-4o-mini"
    assert request.runtime_settings.max_num_concurrent_jurisdictions == 3
    assert request.runtime_settings.log_level == "DEBUG"
    assert request.file_loader_kwargs == {
        "pw_launch_kwargs": {"headless": True, "timeout": 2000}
    }
    if request_class is ExtractionRequest:
        assert (
            request.collection_manifest_fp
            == (tmp_path / "manifest.json").as_posix()
        )


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
