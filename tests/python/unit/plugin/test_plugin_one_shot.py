"""COMPASS one-shot plugin tests"""

from copy import deepcopy
from pathlib import Path

import pytest

from compass.exceptions import COMPASSPluginConfigurationError
from compass.plugin.one_shot.base import (
    _inject_scope_sentinels,
    _normalize_heuristic_keywords,
)
from compass.plugin.one_shot.components import SchemaBasedTextCollector
from compass.validation.content import ParseChunksWithMemory
from compass.warn import COMPASSPluginConfigurationWarning


class _TestSchemaBasedTextCollector(SchemaBasedTextCollector):
    OUT_LABEL = "relevant_text"
    SCHEMA = {
        "$scope": (
            "Ground-source heat pump ordinance provisions, "
            "definitions, and requirements."
        ),
        "type": "object",
        "properties": {
            "technology": {"const": "ground-source heat pump ordinance"}
        },
    }
    CONTENT_VALIDATION_OUTPUT_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["contains_relevant_text", "explanation"],
        "properties": {
            "contains_relevant_text": {"type": "boolean"},
            "explanation": {"type": "string"},
        },
    }
    SCOPE_VALIDATION_OUTPUT_SCHEMA = {
        "type": "object",
        "description": (
            "Response indicating whether the text chunk is within the scope "
            "of the extraction schema."
        ),
        "additionalProperties": False,
        "required": ["matches_scope", "explanation"],
        "properties": {
            "matches_scope": {
                "type": "boolean",
                "description": (
                    "Flag indicating whether the chunk clearly matches the "
                    "schema's intended domain and scope."
                ),
            },
            "explanation": {
                "type": "string",
                "description": (
                    "Short explanation describing why the chunk is in scope "
                    "or out of scope for the extraction schema."
                ),
            },
        },
    }

    def __init__(self, scope_result, context_result):
        self._chunks = {}
        self._scope_result = scope_result
        self._context_result = context_result
        self.response_names = []

    async def call(self, *args, response_format=None, **kwargs):
        """Return canned validation responses keyed by schema name"""
        name = response_format["json_schema"]["name"]
        self.response_names.append(name)
        if name == "chunk_scope_validation":
            return {
                "matches_scope": self._scope_result,
                "explanation": "scope",
            }

        return {
            "contains_relevant_text": self._context_result,
            "explanation": "context",
        }


def test_normalize_heuristic_keywords_normalizes_keys_and_values():
    """Heuristic keyword config should accept flexible keys and values"""

    normalized = _normalize_heuristic_keywords(
        {
            "not tech words": [" County ", None, "county", "  ", "State"],
            "good-tech-keywords": [
                "Wind",
                "Turbine",
                "Setback",
                "Ordinance",
                "wind",
            ],
            "good tech acronyms": ["WEC", "PV", "CUP", 7],
            "GOOD_TECH_PHRASES": [
                "Wind Energy Conversion",
                "Large Wind Energy System",
                "Conditional Use Permit",
            ],
        }
    )

    assert set(normalized) == {
        "NOT_TECH_WORDS",
        "GOOD_TECH_KEYWORDS",
        "GOOD_TECH_ACRONYMS",
        "GOOD_TECH_PHRASES",
    }
    assert set(normalized["NOT_TECH_WORDS"]) == {"county", "state"}
    assert set(normalized["GOOD_TECH_KEYWORDS"]) == {
        "wind",
        "turbine",
        "setback",
        "ordinance",
    }
    assert set(normalized["GOOD_TECH_ACRONYMS"]) == {"wec", "pv", "cup"}
    assert set(normalized["GOOD_TECH_PHRASES"]) == {
        "wind energy conversion",
        "large wind energy system",
        "conditional use permit",
    }


def test_normalize_heuristic_keywords_warns_for_small_good_keyword_set():
    """Small but valid heuristic keyword sets should warn, not fail"""

    with pytest.warns(
        COMPASSPluginConfigurationWarning,
        match="recommended to provide at least 10 total",
    ):
        normalized = _normalize_heuristic_keywords(
            {
                "not_tech_words": ["county"],
                "good_tech_keywords": ["wind"],
                "good_tech_acronyms": ["wec"],
                "good_tech_phrases": ["wind energy conversion"],
            }
        )

    assert set(normalized["GOOD_TECH_KEYWORDS"]) == {"wind"}
    assert set(normalized["GOOD_TECH_ACRONYMS"]) == {"wec"}
    assert set(normalized["GOOD_TECH_PHRASES"]) == {"wind energy conversion"}


def test_normalize_heuristic_keywords_raises_for_missing_required_lists():
    """Heuristic keyword config should require all expected lists"""

    with pytest.raises(
        COMPASSPluginConfigurationError,
        match=r"missing required lists: \['GOOD_TECH_PHRASES'\]",
    ):
        _normalize_heuristic_keywords(
            {
                "not_tech_words": ["county"],
                "good_tech_keywords": ["wind"],
                "good_tech_acronyms": ["wec"],
            }
        )


def test_normalize_heuristic_keywords_raises_for_unexpected_list_name():
    """Heuristic keyword config should reject unknown list names"""

    with pytest.raises(
        COMPASSPluginConfigurationError,
        match=r"Unexpected heuristic keyword list: 'misc_keywords'",
    ):
        _normalize_heuristic_keywords(
            {
                "not_tech_words": ["county"],
                "good_tech_keywords": ["wind"],
                "good_tech_acronyms": ["wec"],
                "good_tech_phrases": ["wind energy conversion"],
                "misc_keywords": ["zoning"],
            }
        )


@pytest.mark.asyncio
async def test_schema_text_collector_short_circuits_when_scope_fails():
    """Collector should stop before relevance check when scope fails"""

    collector = _TestSchemaBasedTextCollector(False, True)
    chunk_parser = ParseChunksWithMemory(["county budgeting text"], 1)

    out = await collector.check_chunk(chunk_parser, 0)

    assert not out
    assert collector.response_names == ["chunk_scope_validation"]
    assert chunk_parser.memory == [{"matches_scope": False}]
    assert not collector._chunks


@pytest.mark.asyncio
async def test_schema_text_collector_stores_chunk_after_scope_and_context():
    """Collector should keep text only after both scope and context pass"""

    collector = _TestSchemaBasedTextCollector(True, True)
    chunk_parser = ParseChunksWithMemory(
        ["header text", "operative geothermal setback text"],
        num_to_recall=2,
    )

    out = await collector.check_chunk(chunk_parser, 1)

    assert out
    assert collector.response_names == [
        "chunk_scope_validation",
        "chunk_validation",
    ]
    assert chunk_parser.memory[1] == {
        "matches_scope": True,
        "contains_relevant_text": True,
    }
    assert collector._chunks == {
        0: "header text",
        1: "operative geothermal setback text",
    }

def _schema_with_scope(enum):
    return {
        "properties": {
            "outputs": {
                "items": {
                    "properties": {"subarea": {"enum": list(enum)}}
                }
            }
        }
    }


def test_inject_subarea_sentinels_wraps_enum():
    """``all`` is prepended and ``other`` appended around user values"""

    schema = _schema_with_subarea(["residential", "commercial"])
    _inject_subarea_sentinels(schema)

    enum = schema["properties"]["outputs"]["items"]["properties"][
        "subarea"
    ]["enum"]
    assert enum == ["all", "residential", "commercial", "other"]


def test_inject_subarea_sentinels_is_idempotent():
    """Calling twice does not duplicate sentinels"""

    schema = _schema_with_subarea(["residential"])
    _inject_subarea_sentinels(schema)
    _inject_subarea_sentinels(schema)

    enum = schema["properties"]["outputs"]["items"]["properties"][
        "subarea"
    ]["enum"]
    assert enum == ["all", "residential", "other"]


def test_inject_subarea_sentinels_noop_when_no_subarea_property():
    """Legacy schemas without a subarea property are left unchanged"""

    schema = {
        "properties": {
            "outputs": {
                "items": {"properties": {"feature": {"enum": ["a", "b"]}}}
            }
        }
    }
    original = deepcopy(schema)

    _inject_subarea_sentinels(schema)

    assert schema == original


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
