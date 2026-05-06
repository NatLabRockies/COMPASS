"""COMPASS ordinance plugin tests"""

import asyncio
from collections import UserList
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from compass.plugin.ordinance import (
    BaseTextCollector,
    BaseTextExtractor,
    BaseParser,
    OrdinanceExtractionPlugin,
)
from compass.exceptions import COMPASSPluginConfigurationError


class MergePlugin(OrdinanceExtractionPlugin):
    """Concrete ordinance plugin for merge tests"""

    TEXT_COLLECTORS = []
    TEXT_EXTRACTORS = []
    PARSERS = []

    IDENTIFIER = "test"
    WEBSITE_KEYWORDS = ["test"]
    QUERY_TEMPLATES = ["test"]
    HEURISTIC = None

    async def parse_docs_for_structured_data(self, extraction_context):
        return extraction_context


class FakeDoc:
    def __init__(self, source, year=None, structured_data=None):
        self.attrs = {"source": source}
        if year is not None:
            self.attrs["date"] = (year, 1, 1)
        if structured_data is not None:
            self.attrs["structured_data"] = structured_data


class FakeExtractionContext(UserList):
    """List-like extraction context for merge tests"""

    def __init__(self, docs):
        super().__init__(docs)
        self.attrs = {}
        self.marked_sources = []

    @property
    def num_documents(self):
        return len(self)

    async def mark_doc_as_data_source(self, doc, out_fn_stem):
        self.marked_sources.append((doc.attrs.get("source"), out_fn_stem))


@pytest.fixture
def merge_plugin():
    """Build a concrete plugin for merge-path tests"""

    plugin = MergePlugin(None, None, None)
    plugin.jurisdiction = SimpleNamespace(full_name="Test County")
    return plugin


def _data_df(*rows):
    return pd.DataFrame(rows)


async def _run_multi_doc_merge(plugin, context, data_dfs):
    """Run the public merge path with controlled per-doc outputs"""

    for doc, data_df in zip(context, data_dfs, strict=True):
        doc.attrs["structured_data"] = data_df

    async def _fake_parse_for_structured_data(doc):
        await asyncio.sleep(0)
        return doc.attrs["structured_data"]

    plugin.parse_for_structured_data = _fake_parse_for_structured_data
    out = await plugin.parse_multi_doc_merge(context)
    return out.attrs["structured_data"]


def test_plugin_validation_parse_key_same():
    """Test plugin interface validation logic"""

    class COLL1(BaseTextCollector):
        OUT_LABEL = "collected"

    class EXT1(BaseTextExtractor):
        IN_LABEL = "collected"
        OUT_LABEL = "extracted"

    class EXT2(BaseTextExtractor):
        IN_LABEL = "collected"
        OUT_LABEL = "extracted_2"

    class PARS1(BaseParser):
        IN_LABEL = "extracted"
        OUT_LABEL = "parsed_1"

    class PARS2(BaseParser):
        IN_LABEL = "collected"
        OUT_LABEL = "parsed_1"

    class MYPlugin(OrdinanceExtractionPlugin):
        TEXT_COLLECTORS = [COLL1]
        TEXT_EXTRACTORS = [EXT1, EXT2]
        PARSERS = [PARS1, PARS2]

        IDENTIFIER = "test"
        WEBSITE_KEYWORDS = ["test"]
        QUERY_TEMPLATES = ["test"]
        HEURISTIC = None

        async def parse_docs_for_structured_data(self, extraction_context):
            return extraction_context

    with pytest.raises(
        COMPASSPluginConfigurationError,
        match="Multiple processing classes produce the same OUT_LABEL key",
    ):
        MYPlugin(None, None, None).validate_plugin_configuration()


def test_plugin_validation_extract_key_same():
    """Test plugin interface validation logic"""

    class COLL1(BaseTextCollector):
        OUT_LABEL = "collected"

    class EXT1(BaseTextExtractor):
        IN_LABEL = "collected"
        OUT_LABEL = "extracted"

    class EXT2(BaseTextExtractor):
        IN_LABEL = "collected"
        OUT_LABEL = "extracted"

    class PARS1(BaseParser):
        IN_LABEL = "extracted"
        OUT_LABEL = "parsed_1"

    class PARS2(BaseParser):
        IN_LABEL = "collected"
        OUT_LABEL = "parsed_2"

    class MYPlugin(OrdinanceExtractionPlugin):
        TEXT_COLLECTORS = [COLL1]
        TEXT_EXTRACTORS = [EXT1, EXT2]
        PARSERS = [PARS1, PARS2]

        IDENTIFIER = "test"
        WEBSITE_KEYWORDS = ["test"]
        QUERY_TEMPLATES = ["test"]
        HEURISTIC = None

        async def parse_docs_for_structured_data(self, extraction_context):
            return extraction_context

    with pytest.raises(
        COMPASSPluginConfigurationError,
        match="Multiple processing classes produce the same OUT_LABEL key",
    ):
        MYPlugin(None, None, None).validate_plugin_configuration()


def test_plugin_validation_no_in_key_for_extract():
    """Test plugin interface validation logic"""

    class COLL1(BaseTextCollector):
        OUT_LABEL = "collected"

    class EXT1(BaseTextExtractor):
        IN_LABEL = "collected"
        OUT_LABEL = "extracted"

    class EXT2(BaseTextExtractor):
        IN_LABEL = "collected_2"
        OUT_LABEL = "extracted_1"

    class PARS1(BaseParser):
        IN_LABEL = "extracted"
        OUT_LABEL = "parsed_1"

    class PARS2(BaseParser):
        IN_LABEL = "collected"
        OUT_LABEL = "parsed_2"

    class MYPlugin(OrdinanceExtractionPlugin):
        TEXT_COLLECTORS = [COLL1]
        TEXT_EXTRACTORS = [EXT1, EXT2]
        PARSERS = [PARS1, PARS2]

        IDENTIFIER = "test"
        WEBSITE_KEYWORDS = ["test"]
        QUERY_TEMPLATES = ["test"]
        HEURISTIC = None

        async def parse_docs_for_structured_data(self, extraction_context):
            return extraction_context

    with pytest.raises(
        COMPASSPluginConfigurationError,
        match=(
            r"One or more processing classes require IN_LABEL 'collected_2', "
            r"which is not produced by any previous processing class: "
            r"\['EXT2'\]"
        ),
    ):
        MYPlugin(None, None, None).validate_plugin_configuration()


def test_plugin_validation_no_in_key_for_parse():
    """Test plugin interface validation logic"""

    class COLL1(BaseTextCollector):
        OUT_LABEL = "collected"

    class EXT1(BaseTextExtractor):
        IN_LABEL = "collected"
        OUT_LABEL = "extracted"

    class EXT2(BaseTextExtractor):
        IN_LABEL = "collected"
        OUT_LABEL = "extracted_1"

    class PARS1(BaseParser):
        IN_LABEL = "extracted"
        OUT_LABEL = "parsed_1"

    class PARS2(BaseParser):
        IN_LABEL = "collected_2"
        OUT_LABEL = "parsed_2"

    class MYPlugin(OrdinanceExtractionPlugin):
        TEXT_COLLECTORS = [COLL1]
        TEXT_EXTRACTORS = [EXT1, EXT2]
        PARSERS = [PARS1, PARS2]

        IDENTIFIER = "test"
        WEBSITE_KEYWORDS = ["test"]
        QUERY_TEMPLATES = ["test"]
        HEURISTIC = None

        async def parse_docs_for_structured_data(self, extraction_context):
            return extraction_context

    with pytest.raises(
        COMPASSPluginConfigurationError,
        match=(
            r"One or more processing classes require IN_LABEL 'collected_2', "
            r"which is not produced by any previous processing class: "
            r"\['PARS2'\]"
        ),
    ):
        MYPlugin(None, None, None).validate_plugin_configuration()


@pytest.mark.asyncio
async def test_merge_multi_doc_data_prefers_latest_year(merge_plugin):
    """Latest dated doc should win overlapping features"""

    context = FakeExtractionContext(
        [
            FakeDoc("older", 2021),
            FakeDoc("newer", 2024),
        ]
    )
    data_dfs = [
        _data_df(
            {"feature": "setback", "value": 100, "summary": "old"},
            {"feature": "height", "value": 80, "summary": "old"},
        ),
        _data_df(
            {"feature": "setback", "value": 150, "summary": "new"},
        ),
    ]

    merged = await _run_multi_doc_merge(merge_plugin, context, data_dfs)

    assert set(merged["feature"].str.casefold()) == {"setback", "height"}
    setback = merged.loc[merged["feature"].str.casefold() == "setback"]
    height = merged.loc[merged["feature"].str.casefold() == "height"]
    assert setback.iloc[0]["value"] == 150
    assert setback.iloc[0]["source"] == "newer"
    assert setback.iloc[0]["year"] == 2024
    assert height.iloc[0]["value"] == 80
    assert height.iloc[0]["source"] == "older"
    assert height.iloc[0]["year"] == 2021
    assert context.marked_sources == [
        ("newer", "Test County_2"),
        ("older", "Test County_1"),
    ]


@pytest.mark.asyncio
async def test_merge_multi_doc_data_falls_back_to_ordinance_count(
    merge_plugin,
):
    """Unknown years should fall back to ordinance count priority"""

    context = FakeExtractionContext(
        [
            FakeDoc("unknown-year"),
            FakeDoc("known-year", 2025),
        ]
    )
    data_dfs = [
        _data_df(
            {"feature": "setback", "value": 100, "summary": "one"},
            {"feature": "height", "value": 50, "summary": "two"},
        ),
        _data_df(
            {"feature": "setback", "value": 200, "summary": "other"},
        ),
    ]

    merged = await _run_multi_doc_merge(merge_plugin, context, data_dfs)

    setback = merged.loc[merged["feature"].str.casefold() == "setback"]
    assert setback.iloc[0]["value"] == 100
    assert setback.iloc[0]["source"] == "unknown-year"
    assert pd.isna(setback.iloc[0]["year"])


@pytest.mark.asyncio
async def test_merge_multi_doc_data_breaks_year_ties_by_row_count(
    merge_plugin,
):
    """Equal years should break ties using ordinance count"""

    context = FakeExtractionContext(
        [
            FakeDoc("fewer", 2024),
            FakeDoc("more", 2024),
        ]
    )
    data_dfs = [
        _data_df(
            {"feature": "setback", "value": 100, "summary": "one"},
        ),
        _data_df(
            {"feature": "setback", "value": 200, "summary": "two"},
            {"feature": "height", "value": 70, "summary": "two"},
        ),
    ]

    merged = await _run_multi_doc_merge(merge_plugin, context, data_dfs)

    setback = merged.loc[merged["feature"].str.casefold() == "setback"]
    assert setback.iloc[0]["value"] == 200
    assert setback.iloc[0]["source"] == "more"


@pytest.mark.asyncio
async def test_merge_multi_doc_data_limits_to_prohibition_documents(
    merge_plugin,
):
    """Any prohibition should limit merging to prohibition docs only"""

    context = FakeExtractionContext(
        [
            FakeDoc("prohibition-older", 2022),
            FakeDoc("prohibition-newer", 2024),
            FakeDoc("non-prohibition", 2026),
        ]
    )
    data_dfs = [
        _data_df(
            {
                "feature": "prohibitions",
                "value": None,
                "summary": "older prohibition",
            },
            {"feature": "height", "value": 90, "summary": "older"},
        ),
        _data_df(
            {
                "feature": "Prohibitions",
                "value": None,
                "summary": "newer prohibition",
            },
            {"feature": "setback", "value": 300, "summary": "newer"},
        ),
        _data_df(
            {"feature": "noise", "value": 45, "summary": "ignored"},
        ),
    ]

    merged = await _run_multi_doc_merge(merge_plugin, context, data_dfs)

    assert set(merged["feature"].str.casefold()) == {
        "prohibitions",
        "setback",
        "height",
    }
    assert "noise" not in set(merged["feature"].str.casefold())
    prohibition = merged.loc[
        merged["feature"].str.casefold() == "prohibitions"
    ]
    assert prohibition.iloc[0]["source"] == "prohibition-newer"
    assert context.marked_sources == [
        ("prohibition-newer", "Test County_2"),
        ("prohibition-older", "Test County_1"),
    ]


@pytest.mark.asyncio
async def test_parse_multi_doc_merge_returns_context(merge_plugin):
    """Public merge path should attach merged structured data"""

    docs = [
        FakeDoc(
            "older",
            2022,
            _data_df(
                {"feature": "height", "value": 60, "summary": "older"},
            ),
        ),
        FakeDoc(
            "newer",
            2024,
            _data_df(
                {"feature": "setback", "value": 100, "summary": "newer"},
            ),
        ),
    ]
    context = FakeExtractionContext(docs)

    async def _fake_parse_for_structured_data(doc):
        await asyncio.sleep(0)
        return doc.attrs["structured_data"]

    merge_plugin.parse_for_structured_data = _fake_parse_for_structured_data

    out = await merge_plugin.parse_multi_doc_merge(context)

    assert out is context
    assert set(out.attrs["structured_data"]["feature"].str.casefold()) == {
        "setback",
        "height",
    }


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
