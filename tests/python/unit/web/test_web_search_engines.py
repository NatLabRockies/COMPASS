"""Tests for DuckDuckGo SerpAPI requests and ELM integration"""

import httpx
import pytest
from elm.web.search import run as elm_search

from compass.pipeline.data_classes import WebSearchParams
from compass.web.search import _run_holistic_sort_search
from compass.web.search_engines import SerpAPIDuckDuckGoSearch


@pytest.fixture
def serpapi_http(monkeypatch):
    """Intercept actual HTTP requests without accessing a live account"""
    requests = []
    settings = []
    reply = {
        "status": 200,
        "body": {
            "organic_results": [
                {
                    "position": 1,
                    "title": "Code",
                    "link": "https://city.gov/code",
                },
                {"position": 2, "title": "No link"},
                {"position": 3, "link": "https://city.gov/ordinance.pdf"},
            ]
        },
    }
    original_client = httpx.AsyncClient

    def respond(request):
        requests.append(request)
        if isinstance(reply["body"], Exception):
            raise reply["body"]
        return httpx.Response(reply["status"], json=reply["body"])

    def client(**kwargs):
        settings.append(kwargs)
        return original_client(
            **kwargs,
            transport=httpx.MockTransport(respond),
        )

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setenv("SERPAPI_KEY", "test-key-not-a-real-secret")
    return requests, settings, reply


@pytest.mark.parametrize("raw", [False, True])
async def test_duckduckgo_request_and_result_shape(serpapi_http, raw):
    """Use the DuckDuckGo endpoint and retain ranking metadata"""
    requests, settings, _ = serpapi_http
    engine = SerpAPIDuckDuckGoSearch(region="uk-en", timeout=25)
    result = await engine._search("city zoning", num_results=2, raw=raw)
    assert dict(requests[0].url.params) == {
        "engine": "duckduckgo",
        "q": "city zoning",
        "kl": "uk-en",
        "api_key": "test-key-not-a-real-secret",
    }
    assert requests[0].url.host == "serpapi.com"
    assert requests[0].url.path == "/search.json"
    assert settings == [{"verify": True, "timeout": 25}]
    if raw:
        assert result[0] == {
            "url": "https://city.gov/code",
            "query": "city zoning",
            "search_engine": "SerpAPI (DuckDuckGo)",
            "query_rank": 1,
            "attrs": {
                "position": 1,
                "title": "Code",
                "link": "https://city.gov/code",
            },
        }
        assert result[1]["query_rank"] == 3
    else:
        assert result == [
            "https://city.gov/code",
            "https://city.gov/ordinance.pdf",
        ]
    assert len(await engine._search("city zoning", num_results=1)) == 1


@pytest.mark.parametrize("body", [{}, {"organic_results": []}])
async def test_no_organic_results(serpapi_http, body):
    """A successful empty search returns an empty result list"""
    serpapi_http[2]["body"] = body
    assert await SerpAPIDuckDuckGoSearch()._search("city") == []


@pytest.mark.parametrize(
    "status,body",
    [
        (401, {"error": "Invalid key"}),
        (429, {"error": "Quota exceeded"}),
        (200, {"error": "Search failed"}),
        (
            200,
            httpx.ReadTimeout("request contained test-key-not-a-real-secret"),
        ),
    ],
)
async def test_failure_follows_elm_empty_result_policy(
    serpapi_http,
    caplog,
    status,
    body,
):
    """ELM logs failures without leaking the key and can try fallbacks"""
    serpapi_http[2].update(status=status, body=body)
    result = await SerpAPIDuckDuckGoSearch().results("city")
    assert result == [[]]
    assert "SerpAPI DuckDuckGo" in caplog.text
    assert "test-key-not-a-real-secret" not in caplog.text


def test_key_required_and_explicit_override(monkeypatch):
    """Resolve the shared key and report missing credentials clearly"""
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    with pytest.raises(ValueError, match="requires SERPAPI_KEY"):
        SerpAPIDuckDuckGoSearch()
    assert SerpAPIDuckDuckGoSearch(api_key="explicit").api_key == "explicit"


async def test_config_merges_api_engines_without_browser(
    serpapi_http,
    monkeypatch,
):
    """Exercise config translation and ELM routing through actual HTTP"""
    kwargs = WebSearchParams(
        search_engines=[
            {"se_name": "SerpAPIGoogleSearch", "verify": False},
            {"se_name": "SerpAPIDuckDuckGoSearch", "verify": False},
        ],
    ).se_kwargs
    assert kwargs["search_engines"] == [
        "SerpAPIGoogleSearch",
        "SerpAPIDuckDuckGoSearch",
    ]
    assert kwargs["duckduckgo_serpapi_kwargs"] == {"verify": False}

    def no_browser(*args, **kwargs):
        pytest.fail("An API search must not launch Playwright")

    monkeypatch.setattr(elm_search, "_single_query_pw", no_browser)
    results = await _run_holistic_sort_search(
        ["city zoning"],
        25,
        None,
        None,
        None,
        "Test City",
        **kwargs,
    )
    assert len(serpapi_http[0]) == 2
    assert {r.url.params["engine"] for r in serpapi_http[0]} == {
        "google",
        "duckduckgo",
    }
    kept = [r for r in results if r["filtered_reason"] is None]
    assert [r["url"] for r in kept] == [
        "https://city.gov/code",
        "https://city.gov/ordinance.pdf",
    ]
    assert all(
        r["duplicates"][0]["search_engine"] == "SerpAPI (DuckDuckGo)"
        for r in kept
    )
    fallback = await elm_search.search_with_fallback(
        ["city zoning"],
        num_urls=2,
        search_engines=["SerpAPIDuckDuckGoSearch"],
    )
    assert set(fallback) == {r["url"] for r in kept}
