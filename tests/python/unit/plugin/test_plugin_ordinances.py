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
    DocSelectionMethod,
    OrdinanceExtractionPlugin,
    _feature_key,
    _fill_in_all_sources,
    _fill_out_multi_file_sources,
    _filter_to_prohibition_cands_if_needed,
    _get_source_inds,
    _has_prohibitions,
    _merge_candidates,
    _prioritize_candidates,
    _valid_chunk,
    _validate_in_out_keys,
)
from compass.exceptions import (
    COMPASSPluginConfigurationError,
    COMPASSRuntimeError,
)


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
async def test_parse_docs_for_structured_data_accepts_enum_value():
    """Enum-valued doc selection should dispatch correctly"""

    class MYPlugin(OrdinanceExtractionPlugin):
        TEXT_COLLECTORS = []
        TEXT_EXTRACTORS = []
        PARSERS = []

        IDENTIFIER = "test"
        WEBSITE_KEYWORDS = ["test"]
        QUERY_TEMPLATES = ["test"]
        HEURISTIC = None
        DOC_SELECTION_METHOD = DocSelectionMethod.MULTI_DOC_ALL

        async def parse_single_doc_for_structured_data(
            self, extraction_context
        ):
            raise AssertionError("wrong dispatch")

        async def parse_multi_doc_context_for_structured_data(
            self, extraction_context
        ):
            raise AssertionError("wrong dispatch")

        async def parse_multi_doc_concat(self, extraction_context):
            return "concat"

        async def parse_multi_doc_merge(self, extraction_context):
            raise AssertionError("wrong dispatch")

    plugin = MYPlugin(None, None, None)

    assert await plugin.parse_docs_for_structured_data(None) == "concat"


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


@pytest.mark.parametrize(
    "chunk,expected",
    [("Useful text", True), ("No relevant text.", False), ("", False)],
)
def test_valid_chunk(chunk, expected):
    """Helper should reject empty and negative extraction responses"""

    assert _valid_chunk(chunk) == expected


def test_validate_in_out_keys_raises_for_missing_key():
    """Helper should fail when no producer satisfies a required input"""

    class Producer:
        OUT_LABEL = "produced"

    class Consumer:
        IN_LABEL = "missing"

    with pytest.raises(
        COMPASSPluginConfigurationError,
        match=r"IN_LABEL 'missing'",
    ):
        _validate_in_out_keys([Consumer], [Producer])


def test_get_source_inds_returns_integer_indices():
    """Helper should extract integer source indices from rows"""

    data_df = _data_df(
        {"feature": "setback", "source": 0},
        {"feature": "height", "source": 1},
        {"feature": "noise", "source": 1},
    )

    source_inds = _get_source_inds(data_df, 3)

    assert list(source_inds) == [0, 1]


@pytest.mark.parametrize(
    "data_df,num_docs,match",
    [
        (_data_df({"feature": "setback"}), 2, "column not found"),
        (
            _data_df({"feature": "setback", "source": "one"}),
            2,
            "non-integer values",
        ),
        (
            _data_df({"feature": "setback", "source": 2}),
            2,
            "out-of-bounds indices",
        ),
    ],
)
def test_get_source_inds_raises_for_invalid_source_values(
    data_df, num_docs, match
):
    """Helper should reject missing, invalid, and out-of-range sources"""

    with pytest.raises(COMPASSRuntimeError, match=match):
        _get_source_inds(data_df, num_docs)


@pytest.mark.asyncio
async def test_fill_out_multi_file_sources_maps_valid_source_indices():
    """Helper should map per-row source indices back to document metadata"""

    context = FakeExtractionContext(
        [FakeDoc("doc-one", 2021), FakeDoc("doc-two", 2024)]
    )
    data_df = _data_df(
        {"feature": "setback", "source": 0},
        {"feature": "height", "source": 1},
    )

    filled = await _fill_out_multi_file_sources(data_df, context, "County")

    assert list(filled["source"]) == ["doc-one", "doc-two"]
    assert list(filled["year"]) == [2021, 2024]
    assert context.marked_sources == [
        ("doc-one", "County_1"),
        ("doc-two", "County_2"),
    ]


@pytest.mark.asyncio
async def test_fill_in_all_sources_reports_full_context_when_needed():
    """Fallback helper should report all documents when row sources fail"""

    context = FakeExtractionContext(
        [FakeDoc("doc-one", 2020), FakeDoc("doc-two", 2024)]
    )
    data_df = _data_df({"feature": "setback", "value": 100})

    filled = await _fill_in_all_sources(data_df, context, "County")

    assert filled.iloc[0]["source"] == "doc-one ;\ndoc-two"
    assert filled.iloc[0]["year"] == 2024
    assert context.marked_sources == [
        ("doc-one", "County_1"),
        ("doc-two", "County_2"),
    ]


def test_feature_key_normalizes_values_and_handles_missing():
    """Feature-key helper should normalize strings and preserve missing"""

    assert _feature_key("  Prohibitions ") == "prohibitions"
    assert _feature_key(pd.NA) is None


def test_has_prohibitions_requires_ordinance_content():
    """Prohibition helper should only flag rows with actual ordinance data"""

    with_prohibition = _data_df(
        {"feature": "Prohibitions", "summary": "Wind is prohibited."}
    )
    without_prohibition = _data_df(
        {"feature": "Prohibitions", "summary": None, "value": None}
    )

    assert _has_prohibitions(with_prohibition)
    assert not _has_prohibitions(without_prohibition)


def test_filter_to_prohibition_candidates_only_when_present():
    """Candidate helper should narrow to prohibition-bearing documents"""

    candidates = [
        {
            "data_df": _data_df(
                {"feature": "setback", "summary": "Regular standard"}
            )
        },
        {
            "data_df": _data_df(
                {
                    "feature": "prohibitions",
                    "summary": "Wind systems are prohibited.",
                }
            )
        },
    ]

    filtered = _filter_to_prohibition_cands_if_needed(candidates)

    assert filtered == [candidates[1]]


def test_prioritize_candidates_prefers_latest_year_then_row_count():
    """Priority helper should sort by year when every candidate has one"""

    candidates = [
        {"year": 2021, "row_count": 5},
        {"year": 2024, "row_count": 1},
        {"year": 2024, "row_count": 3},
    ]

    prioritized = _prioritize_candidates(candidates)

    assert prioritized == [candidates[2], candidates[1], candidates[0]]


def test_prioritize_candidates_falls_back_to_row_count_without_years():
    """Priority helper should ignore year sorting when any year is unknown"""

    candidates = [
        {"year": 2024, "row_count": 1},
        {"year": None, "row_count": 3},
        {"year": 2021, "row_count": 2},
    ]

    prioritized = _prioritize_candidates(candidates)

    assert prioritized == [candidates[1], candidates[2], candidates[0]]


@pytest.mark.asyncio
async def test_merge_candidates_keeps_first_feature_and_marks_sources():
    """Merge helper should keep first-seen features by candidate priority"""

    context = FakeExtractionContext([FakeDoc("older"), FakeDoc("newer")])
    candidates = [
        {
            "data_df": _data_df(
                {"feature": "setback", "value": 200, "source": "newer"},
                {"feature": "height", "value": 80, "source": "newer"},
            ),
            "doc": context[1],
            "doc_ind": 2,
        },
        {
            "data_df": _data_df(
                {"feature": "setback", "value": 100, "source": "older"},
                {"feature": "noise", "value": 45, "source": "older"},
            ),
            "doc": context[0],
            "doc_ind": 1,
        },
    ]

    merged = await _merge_candidates(candidates, context, "County")

    assert set(merged["feature"].str.casefold()) == {
        "setback",
        "height",
        "noise",
    }
    setback = merged.loc[merged["feature"].str.casefold() == "setback"]
    assert setback.iloc[0]["value"] == 200
    assert context.marked_sources == [
        ("newer", "County_2"),
        ("older", "County_1"),
    ]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
