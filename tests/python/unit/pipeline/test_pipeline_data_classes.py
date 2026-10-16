"""Test COMPASS Ordinance logging logic"""

from pathlib import Path

import pytest

from compass.exceptions import COMPASSValueError
from compass.pipeline import (
    BaseRequest,
    CollectionRequest,
    ExtractionRequest,
    ProcessRequest,
    SearchRequest,
)
from compass.pipeline.runtime import PipelineRuntime
from compass.pipeline.data_classes import (
    RuntimeSettings,
    WebSearchParams,
    build_models,
)
from compass.llm import LiteLLMConfig, OpenAIConfig
from compass.utilities.io import ConfigType, load_config
from compass.warn import COMPASSWarning


@pytest.mark.parametrize("limit", [0, -1, 1.0, 1.5, "2", None, True, False])
def test_runtime_settings_rejects_invalid_jurisdiction_concurrency(limit):
    """Reject jurisdiction concurrency that is not a positive integer"""
    with pytest.raises(
        COMPASSValueError,
        match="max_num_concurrent_jurisdictions must be an integer >= 1",
    ):
        RuntimeSettings(max_num_concurrent_jurisdictions=limit)


@pytest.mark.parametrize("limit", [1, 3, 25])
def test_runtime_settings_accepts_positive_jurisdiction_concurrency(limit):
    """Preserve valid jurisdiction concurrency limits"""
    settings = RuntimeSettings(max_num_concurrent_jurisdictions=limit)
    assert settings.max_num_concurrent_jurisdictions == limit


@pytest.mark.parametrize(
    "request_class",
    [
        BaseRequest,
        ProcessRequest,
        CollectionRequest,
        SearchRequest,
        ExtractionRequest,
    ],
)
def test_requests_reject_zero_jurisdiction_concurrency(
    tmp_path, request_class
):
    """Validate shared runtime settings in every request type"""
    extra_config = (
        {"collection_manifest_fp": "./manifest.json"}
        if request_class is ExtractionRequest
        else {}
    )
    with pytest.raises(
        COMPASSValueError,
        match="max_num_concurrent_jurisdictions must be an integer >= 1",
    ):
        request_class(
            out_dir=tmp_path,
            tech="solar",
            jurisdiction_fp="jurisdictions.csv",
            max_num_concurrent_jurisdictions=0,
            **extra_config,
        )


@pytest.mark.parametrize("config_type", list(ConfigType))
@pytest.mark.parametrize(
    "request_class, source_dir_key, extra_config",
    [
        (BaseRequest, "ordinance_file_dir", {}),
        (ProcessRequest, "ordinance_file_dir", {}),
        (CollectionRequest, "source_file_dir", {}),
        (SearchRequest, "log_dir", {}),
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
    is_search = request_class is SearchRequest
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
            "log_level": "WARNING",
            **(
                {
                    "model": "gpt-4o-mini",
                    "max_num_concurrent_jurisdictions": 3,
                    "file_loader_kwargs": {
                        "pw_launch_kwargs": {
                            "headless": True,
                            "timeout": 1000,
                        }
                    },
                }
                if not is_search
                else {}
            ),
        },
    )
    config_type.write(
        child,
        {
            "inherit_from": "parents/parent.json",
            "out_dir": "./child_outputs",
            "log_level": "DEBUG",
            **(
                {
                    "num_search_results_to_crawl": 3,
                    "search_results_crawl_depth": 2,
                }
                if request_class not in {ExtractionRequest, SearchRequest}
                else {}
            ),
            **(
                {"file_loader_kwargs": {"pw_launch_kwargs": {"timeout": 2000}}}
                if not is_search
                else {}
            ),
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
        request.output_settings.log_dir
        if is_search
        else request.output_settings.ordinance_file_dir
    ) == (parent_dir / "sources").as_posix()
    assert request.user_model_input == (None if is_search else "gpt-4o-mini")
    assert request.runtime_settings.max_num_concurrent_jurisdictions == (
        25 if is_search else 3
    )
    assert request.runtime_settings.log_level == "DEBUG"
    if request_class not in {ExtractionRequest, SearchRequest}:
        assert request.search_settings.num_search_results_to_crawl == 3
        assert request.search_settings.search_results_crawl_depth == 2
    assert request.file_loader_kwargs == (
        None
        if is_search
        else {"pw_launch_kwargs": {"headless": True, "timeout": 2000}}
    )
    if request_class is ExtractionRequest:
        assert (
            request.collection_manifest_fp
            == (tmp_path / "manifest.json").as_posix()
        )


@pytest.mark.parametrize("request_class", [CollectionRequest, ProcessRequest])
def test_saved_search_request(tmp_path, request_class):
    """Forward saved search inputs through both consumer requests"""
    manifest_fp = tmp_path / "search_result_manifest.json"
    request = request_class(
        tmp_path / "output",
        "wind",
        None,
        search_result_manifest_fp=manifest_fp,
    )
    assert request.search_result_manifest_fp == manifest_fp


def test_search_runtime_resources(tmp_path):
    """Search needs no models, document workers, or document directories"""
    request = SearchRequest(tmp_path / "output", "wind", None)
    runtime = PipelineRuntime(request)
    assert request.user_model_input is None
    assert request.output_settings.save_search_engine_results is True
    assert runtime.models == {}
    assert [type(service).__name__ for service in runtime._services] == [
        "GenericFuncRunner"
    ]
    assert {path.name for path in runtime.dirs.out.iterdir()} == {
        "logs",
        "se_results",
    }


@pytest.mark.parametrize(
    "option",
    [
        "model",
        "ordinance_file_dir",
        "num_search_results_to_crawl",
        "file_loader_kwargs",
        "save_search_engine_results",
    ],
)
def test_search_request_rejects_non_search_options(tmp_path, option):
    """Search requests reject settings for collection or extraction"""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        SearchRequest(tmp_path / "output", "wind", None, **{option: None})


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


@pytest.mark.parametrize(
    "model_input,config_type,client_type",
    [
        ("gpt-4o-mini", OpenAIConfig, "azure"),
        ("openai/gpt-5-mini", COMPASSValueError, None),
        ([{"name": "gpt-4o-mini"}], OpenAIConfig, "azure"),
        (
            [{"name": "gpt-4o-mini", "client_type": "openai"}],
            OpenAIConfig,
            "openai",
        ),
        (
            [{"name": "anthropic/claude-sonnet-4-5"}],
            COMPASSValueError,
            None,
        ),
        (
            [{"name": "anthropic/claude-sonnet-4-5", "client_type": "litellm"}],
            LiteLLMConfig,
            "litellm",
        ),
        (
            [{"name": "my-model", "client_type": "LiteLLM"}],
            LiteLLMConfig,
            "litellm",
        ),
        (
            [{"name": "org/model", "client_type": "openai"}],
            OpenAIConfig,
            "openai",
        ),
    ],
)
def test_build_models_routes_client_type(
    model_input, config_type, client_type
):
    """Explicit client type controls routing and Azure stays default"""
    if config_type is COMPASSValueError:
        with pytest.raises(
            COMPASSValueError,
            match="require an explicit client_type|must include an explicit provider",
        ):
            build_models(model_input)
        return

    config = build_models(model_input)["default"]

    assert type(config) is config_type
    assert config.client_type == client_type


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


def test_search_crawl_settings(tmp_path):
    """Search crawl is opt-in and request settings preserve zero depth"""
    defaults = WebSearchParams()
    assert defaults.num_search_results_to_crawl == 0
    assert defaults.search_results_crawl_depth == 3
    request = CollectionRequest(
        out_dir=tmp_path,
        tech="solar",
        jurisdiction_fp="jurisdictions.csv",
        num_urls_to_check_per_jurisdiction=10,
        num_search_results_to_crawl=7,
        search_results_crawl_depth=0,
    )
    assert request.search_settings.num_search_results_to_crawl == 7
    assert request.search_settings.search_results_crawl_depth == 0


def test_search_crawl_budget_warns_and_caps_at_url_limit():
    """Excess crawl budgets warn and use the direct URL limit"""
    with pytest.warns(COMPASSWarning) as warning_records:
        settings = WebSearchParams(
            num_urls_to_check_per_jurisdiction=5,
            num_search_results_to_crawl=7,
        )

    assert len(warning_records) == 1
    assert str(warning_records[0].message) == (
        "Number of ranked search results to crawl (7) exceeds the "
        "number of unique search result URLs to check for each "
        "jurisdiction (5); using 5"
    )
    assert settings.num_search_results_to_crawl == 5
    assert settings.num_urls_to_check_per_jurisdiction == 5


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
