"""COMPASS search adapters registered with ELM's search orchestration"""

import httpx
from elm.web.search.base import (
    APISearchEngineLinkSearch,
    format_search_results,
)
from elm.web.search.run import SEARCH_ENGINE_OPTIONS


class SerpAPIDuckDuckGoSearch(APISearchEngineLinkSearch):
    """Retrieve DuckDuckGo organic results through SerpAPI

    Parameters
    ----------
    api_key : str, optional
        SerpAPI key. Defaults to ``SERPAPI_KEY`` in the environment,
        shared with ELM's Google SerpAPI adapter.
    verify : bool, default=True
        Verify the API server's TLS certificate.
    region : str, default="us-en"
        DuckDuckGo region, sent as SerpAPI's ``kl`` parameter.
    timeout : float, default=60
        HTTP request timeout in seconds.

    Notes
    -----
    Uses https://serpapi.com/duckduckgo-search-api. ELM schedules this
    adapter through its API path, without launching a search browser.
    """

    _SE_NAME = "SerpAPI (DuckDuckGo)"
    API_KEY_VAR = "SERPAPI_KEY"

    def __init__(self, api_key=None, verify=True, region="us-en", timeout=60):
        super().__init__(api_key=api_key)
        if not self.api_key:
            msg = "SerpAPI DuckDuckGo search requires SERPAPI_KEY or api_key"
            raise ValueError(msg)
        self.verify = verify
        self.region = region
        self.timeout = timeout

    async def _search(self, query, num_results=10, raw=False):
        """Return URLs or ELM records for one query"""
        params = {
            "engine": "duckduckgo",
            "q": query,
            "kl": self.region,
            "api_key": self.api_key,
        }
        try:
            async with httpx.AsyncClient(
                verify=self.verify,
                timeout=self.timeout,
            ) as client:
                response = await client.get(
                    "https://serpapi.com/search.json",
                    params=params,
                )
                response.raise_for_status()
        except httpx.HTTPError:
            # HTTP exceptions include the request URL (and API key).
            # ELM logs exceptions, so expose only a sanitized message.
            msg = "SerpAPI DuckDuckGo HTTP request failed"
            raise RuntimeError(msg) from None

        payload = response.json()
        if payload.get("error"):
            msg = "SerpAPI DuckDuckGo returned an API error"
            raise RuntimeError(msg)
        return format_search_results(
            self._SE_NAME,
            query,
            payload.get("organic_results") or [],
            url_key="link",
            raw=raw,
        )[:num_results]


# Extend the shared registry so both config parsing and ELM's search,
# fallback, and download functions resolve the same adapter. Reuse the
# existing entry type without importing ELM's private namedtuple class.
SEARCH_ENGINE_OPTIONS.setdefault(
    "SerpAPIDuckDuckGoSearch",
    SEARCH_ENGINE_OPTIONS["SerpAPIGoogleSearch"]._replace(
        se_class=SerpAPIDuckDuckGoSearch,
        uses_browser=False,
        kwg_key_name="duckduckgo_serpapi_kwargs",
    ),
)
