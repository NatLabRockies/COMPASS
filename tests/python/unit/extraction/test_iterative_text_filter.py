"""Unit tests for focused text filtering"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from compass.extraction.iterative_text_filter import (
    FocusedTextFilter,
    _extract_keywords,
)


@pytest.fixture
def text_filter():
    """Create text filter with keyword strategy"""
    return FocusedTextFilter(strategy="keyword", context_window=1)


@pytest.fixture
def sample_text():
    """Sample ordinance text"""
    return (
        "Section 1: General provisions. The purpose of this ordinance is to "
        "regulate solar energy systems in the county. "
        * 5
        + "Section 2: Setback requirements. Solar energy systems must be "
        "setback at least 50 feet from property lines. Systems must also "
        "maintain 100 feet from structures. "
        * 3
        + "Section 3: Other provisions. Decommissioning requirements apply. "
        "Visual impact assessments required. " * 5
    )


@pytest.fixture
def sample_schema():
    """Sample extraction schema"""
    return {
        "property_line": {
            "description": "Setback distance from property line boundaries"
        },
        "structures": {
            "description": "Setback distance from structures and buildings"
        },
        "decommissioning": {
            "description": "Decommissioning requirements for removal"
        },
    }


def test_keyword_extraction(text_filter, sample_schema):
    """Test keyword extraction from feature description"""
    keywords = _extract_keywords("property_line", sample_schema)

    assert isinstance(keywords, list)
    assert len(keywords) > 0
    assert any("property" in kw.lower() for kw in keywords)


def test_keyword_extraction_from_feature_name(text_filter):
    """Test keyword extraction from feature name alone"""
    keywords = _extract_keywords("property_line", {})

    assert "property" in keywords
    assert "line" in keywords


@pytest.mark.asyncio
async def test_filter_for_features(text_filter, sample_text, sample_schema):
    """Test filtering text for specific features"""
    result = await text_filter.filter_for_features(
        text=sample_text,
        feature_list=["property_line", "structures"],
        schema=sample_schema,
    )

    assert isinstance(result, dict)
    assert "property_line" in result
    assert "structures" in result

    assert result["property_line"]
    assert (
        "property" in result["property_line"].lower()
        or "setback" in result["property_line"].lower()
    )


@pytest.mark.asyncio
async def test_filtered_text_contains_relevant_info(
    text_filter, sample_text, sample_schema
):
    """Test that filtered text contains relevant information"""
    result = await text_filter.filter_for_features(
        text=sample_text,
        feature_list=["property_line"],
        schema=sample_schema,
    )

    filtered = result["property_line"]

    assert "50 feet" in filtered or "property" in filtered.lower()


@pytest.mark.asyncio
async def test_filtered_text_includes_context(sample_text, sample_schema):
    """Test that context window includes surrounding chunks"""
    filter_with_context = FocusedTextFilter(
        strategy="keyword", context_window=2
    )

    result = await filter_with_context.filter_for_features(
        text=sample_text,
        feature_list=["property_line"],
        schema=sample_schema,
    )

    filter_without_context = FocusedTextFilter(
        strategy="keyword", context_window=0
    )

    result_no_context = await filter_without_context.filter_for_features(
        text=sample_text,
        feature_list=["property_line"],
        schema=sample_schema,
    )

    assert len(result["property_line"]) >= len(
        result_no_context["property_line"]
    )


def test_context_window_expansion(text_filter):
    """Test context window expansion logic"""
    indices = [5]
    total_chunks = 10

    expanded = text_filter._expand_with_context(indices, total_chunks)

    assert 5 in expanded
    assert 4 in expanded
    assert 6 in expanded
    assert len(expanded) >= 3


def test_context_window_respects_boundaries(text_filter):
    """Test context window doesn't exceed chunk boundaries"""
    indices = [0]
    total_chunks = 10

    expanded = text_filter._expand_with_context(indices, total_chunks)

    assert 0 in expanded
    assert 1 in expanded
    assert -1 not in expanded

    indices = [9]
    expanded = text_filter._expand_with_context(indices, total_chunks)

    assert 9 in expanded
    assert 8 in expanded
    assert 10 not in expanded


@pytest.mark.asyncio
async def test_missing_feature_returns_full_text(
    text_filter, sample_text, sample_schema
):
    """Test that missing features return full text as fallback"""
    result = await text_filter.filter_for_features(
        text=sample_text,
        feature_list=["nonexistent_feature"],
        schema=sample_schema,
    )

    assert "nonexistent_feature" in result
    assert result["nonexistent_feature"] == sample_text


@pytest.mark.asyncio
async def test_semantic_search_returns_matching_indices(sample_schema):
    """Test semantic search returns indices marked relevant"""
    semantic_filter = FocusedTextFilter(
        strategy="semantic",
        llm_service=AsyncMock(),
        context_window=0,
    )

    semantic_filter._check_chunk_relevance = AsyncMock(
        side_effect=[
            {
                "is_relevant": False,
                "confidence": 0.9,
                "reason": "Not about setbacks",
            },
            {
                "is_relevant": True,
                "confidence": 0.95,
                "reason": "Contains setback requirement",
            },
            {
                "is_relevant": False,
                "confidence": 0.8,
                "reason": "Unrelated section",
            },
        ]
    )

    indices = await semantic_filter._semantic_search(
        ["Chunk A", "Chunk B", "Chunk C"],
        "property_line",
        sample_schema,
    )

    assert indices == [1]


@pytest.mark.asyncio
async def test_semantic_search_malformed_response_fallback(sample_schema):
    """Test semantic search falls back to keyword on bad schema output"""
    semantic_filter = FocusedTextFilter(
        strategy="semantic",
        llm_service=AsyncMock(),
        context_window=0,
    )
    semantic_filter._check_chunk_relevance = AsyncMock(
        return_value={
            "confidence": 0.2,
            "reason": "Missing required boolean",
        }
    )

    with patch(
        "compass.extraction.iterative_text_filter._keyword_search"
    ) as mock_keyword:
        mock_keyword.return_value = [0]
        indices = await semantic_filter._semantic_search(
            ["Chunk A"],
            "property_line",
            sample_schema,
        )

    assert indices == [0]


@pytest.mark.asyncio
async def test_semantic_search_exception_fallback(sample_schema):
    """Test semantic search falls back to keyword on LLM errors"""
    semantic_filter = FocusedTextFilter(
        strategy="semantic",
        llm_service=AsyncMock(),
        context_window=0,
    )
    semantic_filter._check_chunk_relevance = AsyncMock(
        side_effect=Exception("LLM request failed")
    )

    with patch(
        "compass.extraction.iterative_text_filter._keyword_search"
    ) as mock_keyword:
        mock_keyword.return_value = [2]
        indices = await semantic_filter._semantic_search(
            ["Chunk A", "Chunk B", "Chunk C"],
            "property_line",
            sample_schema,
        )

    assert indices == [2]


@pytest.mark.asyncio
async def test_semantic_without_llm_service_fallback(sample_schema):
    """Test semantic strategy falls back to keyword without LLM service"""
    semantic_filter = FocusedTextFilter(
        strategy="semantic",
        llm_service=None,
        context_window=0,
    )

    with patch(
        "compass.extraction.iterative_text_filter._keyword_search"
    ) as mock_keyword:
        mock_keyword.return_value = [1]
        indices = await semantic_filter._semantic_search(
            ["Chunk A", "Chunk B"],
            "property_line",
            sample_schema,
        )

    assert indices == [1]


@pytest.mark.asyncio
async def test_hybrid_combines_semantic_and_keyword(sample_schema):
    """Test hybrid strategy unions semantic and keyword results"""
    hybrid_filter = FocusedTextFilter(
        strategy="hybrid",
        llm_service=AsyncMock(),
        context_window=0,
    )
    hybrid_filter._splitter.split_text = MagicMock(
        return_value=["Chunk A", "Chunk B", "Chunk C"]
    )
    hybrid_filter._semantic_search = AsyncMock(return_value=[1])

    with patch(
        "compass.extraction.iterative_text_filter._keyword_search"
    ) as mock_keyword:
        mock_keyword.return_value = [0, 1]
        result = await hybrid_filter.filter_for_features(
            text="ignored",
            feature_list=["property_line"],
            schema=sample_schema,
        )

    assert "Chunk A" in result["property_line"]
    assert "Chunk B" in result["property_line"]
    assert "Chunk C" not in result["property_line"]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
