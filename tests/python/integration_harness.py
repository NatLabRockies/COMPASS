"""Reusable offline integration harness for COMPASS"""

import hashlib
import json
import socket
from csv import DictWriter
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
        self._validate()
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
        original_connect = socket.socket.connect

        def guarded_connect(sock, address):
            host = address[0] if isinstance(address, tuple) else None
            is_network_socket = sock.family in {
                socket.AF_INET,
                socket.AF_INET6,
            }
            is_localhost = host in {
                "127.0.0.1",
                "::1",
                "localhost",
            }
            if is_network_socket and not is_localhost:
                msg = (
                    "Unexpected network connection in offline test: "
                    f"{address}"
                )
                raise AssertionError(msg)
            return original_connect(sock, address)

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
        monkeypatch.setattr(socket.socket, "connect", guarded_connect)
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

    def write_jurisdictions(self, output_fp):
        """Write the scenario jurisdiction as a pipeline CSV input"""
        jurisdiction = self.config["jurisdiction"]
        field_map = {
            "State": "state",
            "County": "county",
            "Subdivision": "subdivision",
            "Jurisdiction Type": "jurisdiction_type",
            "FIPS": "fips",
            "Website": "website",
        }
        output_fp = Path(output_fp)
        with output_fp.open("w", encoding="utf-8", newline="") as file:
            writer = DictWriter(file, fieldnames=field_map)
            writer.writeheader()
            writer.writerow(
                {
                    heading: jurisdiction.get(key) or ""
                    for heading, key in field_map.items()
                }
            )
        return output_fp

    def collection_request_kwargs(self, out_dir, tech):
        """Build collection request inputs from scenario data"""
        jurisdiction = self.config["jurisdiction"]
        settings = self.config.get("settings", {})
        kwargs = {
            "out_dir": out_dir,
            "tech": tech,
            "perform_se_search": settings.get(
                "perform_se_search", "search_engine" in self.config
            ),
            "perform_website_search": settings.get(
                "perform_website_search",
                any(
                    channel in self.config
                    for channel in (
                        "elm_website_crawl",
                        "compass_website_crawl",
                    )
                ),
            ),
        }
        if "known_urls" in self.config:
            kwargs["known_doc_urls"] = {
                str(jurisdiction["fips"]): [
                    {"source": item["source"]}
                    for item in self.config["known_urls"]
                ]
            }
        return kwargs

    def process_config(self, out_dir, tech, jurisdiction_fp):
        """Build a serializable process CLI configuration"""
        config = self.collection_request_kwargs(out_dir, tech)
        config["out_dir"] = str(config["out_dir"])
        config["jurisdiction_fp"] = str(jurisdiction_fp)
        config["model"] = "offline-replay"
        return config

    @property
    def expected_document_count(self):
        """int: Number of documents configured across active channels"""
        return sum(
            len(self.config.get(channel, [])) for channel in self.CHANNELS
        )

    def assert_consumed(self):
        """Assert all configured external interactions occurred"""
        expected = Counter(
            {
                channel: self.config.get("expected_calls", {}).get(channel, 1)
                for channel in self.CHANNELS
                if channel in self.config
            }
        )
        expected["redirect"] = self.config.get("expected_calls", {}).get(
            "redirect",
            sum(
                channel.endswith("website_crawl")
                for channel in self.config
                if channel in self.CHANNELS
            ),
        )
        expected += Counter()
        assert self.calls == expected, (
            f"External call mismatch: actual={self.calls!r}, "
            f"expected={expected!r}"
        )
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
        expected = self.config.get(
            "website", self.config["jurisdiction"].get("website")
        )
        assert website.rstrip("/") == expected.rstrip("/")

    def _validate(self):
        """Validate required and coupled scenario fields"""
        jurisdiction = self.config.get("jurisdiction", {})
        required = {"state", "jurisdiction_type", "fips"}
        missing = required - jurisdiction.keys()
        if missing:
            raise ValueError(
                f"Scenario jurisdiction is missing fields: {sorted(missing)}"
            )

        crawl_channels = {
            "elm_website_crawl",
            "compass_website_crawl",
        }
        configured_crawls = crawl_channels & self.config.keys()
        if configured_crawls and configured_crawls != crawl_channels:
            raise ValueError(
                "Website scenarios must configure both crawl channels; "
                "use an empty list when one should return no documents"
            )

        settings = self.config.get("settings", {})
        website_search = settings.get(
            "perform_website_search", bool(configured_crawls)
        )
        if website_search and not configured_crawls:
            raise ValueError(
                "Website search requires both crawl channel fixtures"
            )
        if not website_search and configured_crawls:
            raise ValueError(
                "Website crawl fixtures require website search to be enabled"
            )
        if configured_crawls and not (
            self.config.get("website") or jurisdiction.get("website")
        ):
            raise ValueError(
                "Website crawl fixtures require a jurisdiction website"
            )

        search_configured = "search_engine" in self.config
        search_enabled = settings.get(
            "perform_se_search", search_configured
        )
        if search_enabled != search_configured:
            raise ValueError(
                "Search engine settings and fixtures must be enabled together"
            )
