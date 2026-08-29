"""Reusable offline integration harness for COMPASS"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from elm.web.document import HTMLDocument

import compass.pipeline.collection.steps as collection_steps
import compass.pipeline.data_classes as data_classes
from compass.services.base import Service
from compass.utilities.enums import LLMTasks


class ReplayLLMService(Service):
    """Strict queue-backed LLM replay service"""

    def __init__(self, responses):
        self.responses = [dict(response) for response in responses]
        self.calls = []

    @property
    def can_process(self):
        """bool: Always ready to process"""
        return True

    async def process(self, *args, **kwargs):
        """Return the response matching the recorded prompt"""
        messages = kwargs.get("messages", [])
        prompt = "\n".join(message.get("content", "") for message in messages)
        matches = [
            response
            for response in self.responses
            if not response.get("_used")
            and response["prompt_contains"] in prompt
        ]
        if len(matches) != 1:
            expected = [
                response["prompt_contains"]
                for response in self.responses
                if not response.get("_used")
            ]
            msg = (
                "Expected exactly one replay response for prompt "
                f"{prompt!r}; unmatched responses: {expected!r}"
            )
            raise AssertionError(msg)

        match = matches[0]
        match["_used"] = True
        self.calls.append(
            {
                "messages": messages,
                "usage_sub_label": kwargs.get("usage_sub_label"),
            }
        )
        return match["response"]

    def assert_consumed(self):
        """Assert that every configured LLM response was used"""
        unused = [
            response["prompt_contains"]
            for response in self.responses
            if not response.get("_used")
        ]
        assert not unused, f"Unused LLM replay responses: {unused!r}"


class ReplayModelConfig:
    """Minimal model configuration backed by replayed responses"""

    def __init__(self, service):
        self.name = "offline-replay-model"
        self.llm_service = service
        self.llm_call_kwargs = {}
        self.llm_service_rate_limit = 1_000_000
        self.text_splitter_chunk_size = 10_000
        self.text_splitter_chunk_overlap = 0
        self.client_type = "offline-replay"


class OfflineScenario:
    """Strict fixture-driven replacements for external pipeline services"""

    CHANNELS = (
        "known_urls",
        "search_engine",
        "elm_website_crawl",
        "compass_website_crawl",
    )

    def __init__(self, config, cache_dir):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = Counter()
        self.llm_service = ReplayLLMService(config.get("llm_responses", []))

    @classmethod
    def from_file(cls, scenario_fp, cache_dir):
        """Load a replay scenario from JSON"""
        config = json.loads(Path(scenario_fp).read_text(encoding="utf-8"))
        return cls(config, cache_dir)

    def install(self, monkeypatch):
        """Patch every network and LLM boundary used by collection"""

        def build_models(*args, **kwargs):
            return {
                LLMTasks.DEFAULT: ReplayModelConfig(self.llm_service),
            }

        async def download_known_urls(jurisdiction, urls, **kwargs):
            self.calls["known_urls"] += 1
            expected = [
                item["source"] for item in self.config["known_urls"]
            ]
            assert urls == expected
            return self._documents("known_urls")

        async def download_from_search(*args, **kwargs):
            self.calls["search_engine"] += 1
            return self._documents("search_engine")

        async def download_from_elm(website, **kwargs):
            self.calls["elm_website_crawl"] += 1
            self._assert_website(website)
            return self._documents("elm_website_crawl"), []

        async def download_from_compass(website, **kwargs):
            self.calls["compass_website_crawl"] += 1
            self._assert_website(website)
            return self._documents("compass_website_crawl")

        async def redirected_url(url, **kwargs):
            self.calls["redirect"] += 1
            self._assert_website(url)
            return url

        monkeypatch.setattr(data_classes, "build_models", build_models)
        monkeypatch.setattr(
            collection_steps, "download_known_urls", download_known_urls
        )
        monkeypatch.setattr(
            collection_steps,
            "download_jurisdiction_ordinance_using_search_engine",
            download_from_search,
        )
        monkeypatch.setattr(
            collection_steps,
            "download_jurisdiction_ordinances_from_website",
            download_from_elm,
        )
        monkeypatch.setattr(
            collection_steps,
            "download_jurisdiction_ordinances_from_website_compass_crawl",
            download_from_compass,
        )
        monkeypatch.setattr(
            collection_steps, "get_redirected_url", redirected_url
        )

    def assert_consumed(self):
        """Assert all configured external interactions occurred"""
        expected = {
            channel: 1
            for channel in self.CHANNELS
            if channel in self.config
        }
        actual = {channel: self.calls[channel] for channel in expected}
        assert actual == expected
        if any(
            channel.endswith("website_crawl") for channel in self.config
        ):
            assert self.calls["redirect"] == 1
        self.llm_service.assert_consumed()

    def _documents(self, channel):
        """Build replay documents for one collection channel"""
        return [self._document(item) for item in self.config.get(channel, [])]

    def _document(self, item):
        """Build one file-backed HTML document"""
        source = item["source"]
        content = item["content"]
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        cache_fp = self.cache_dir / f"{digest}.html"
        cache_fp.write_text(content, encoding="utf-8")
        return HTMLDocument(
            [content],
            attrs={
                "source": source,
                "cache_fn": cache_fp,
                "checksum": f"sha256:{digest}",
                "doc_type": "html",
            },
        )

    def _assert_website(self, website):
        """Assert a website call matches the scenario"""
        assert website.rstrip("/") == self.config["website"].rstrip("/")

