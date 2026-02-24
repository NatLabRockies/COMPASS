"""Test iterative extraction graph"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from compass.extraction.iterative import IterativeExtractionGraph


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for graph tests"""
    service = AsyncMock()
    service.call = AsyncMock()
    return service


@pytest.fixture
def mock_usage_tracker():
    """Mock usage tracker for graph tests"""
    tracker = MagicMock()
    tracker.track_usage = MagicMock()
    return tracker


@pytest.fixture
def mock_parser():
    """Mock parser for graph tests"""
    parser = AsyncMock()
    parser.reextract_focused_features = AsyncMock(return_value={})
    return parser


@pytest.fixture
def sample_config():
    """Sample configuration"""
    return {
        "max_iterations": 5,
        "validation_strictness": "moderate",
        "text_filter_strategy": "hybrid",
        "merge_strategy": "replace",
        "context_window": 2,
    }


@pytest.fixture
def sample_extraction():
    """Sample extraction result"""
    return {
        "property_line": {"value": 100, "units": "feet"},
        "structures": {"value": None, "units": None},
    }


@pytest.fixture
def sample_schema():
    """Sample schema"""
    return {
        "property_line": "Distance to property line",
        "structures": "Distance to structures",
    }


def test_graph_initialization(mock_llm_service, sample_config):
    """Test graph initializes correctly"""
    graph = IterativeExtractionGraph(
        config=sample_config, llm_service=mock_llm_service
    )
    assert graph is not None
    assert graph._workflow is not None


@pytest.mark.asyncio
async def test_run_already_valid(
    mock_llm_service, mock_parser, sample_config, sample_extraction, sample_schema
):
    """Test workflow when initial extraction is already valid"""
    with patch(
        "compass.extraction.iterative.ExtractionValidator"
    ) as mock_validator_class:
        mock_validator = AsyncMock()
        mock_validator.validate = AsyncMock(
            return_value={"is_valid": True, "issues": []}
        )
        mock_validator_class.return_value = mock_validator

        graph = IterativeExtractionGraph(
            config=sample_config, llm_service=mock_llm_service
        )

        complete_extraction = {
            "property_line": {"value": 100, "units": "feet"},
            "structures": {"value": 50, "units": "feet"},
        }

        result = await graph.run(
            text="Sample ordinance text",
            schema=sample_schema,
            initial_extraction=complete_extraction,
            parser=mock_parser,
        )

        assert result["metadata"]["iterations"] == 0
        assert result["metadata"]["final_valid"] is True
        mock_parser.reextract_focused_features.assert_not_called()


@pytest.mark.asyncio
async def test_run_single_iteration(
    mock_llm_service, mock_parser, sample_config, sample_extraction, sample_schema
):
    """Test workflow with single iteration"""
    with patch(
        "compass.extraction.iterative.ExtractionValidator"
    ) as mock_validator_class, patch(
        "compass.extraction.iterative.FocusedTextFilter"
    ) as mock_filter_class:
        mock_validator = AsyncMock()
        mock_validator.validate = AsyncMock()
        mock_validator.validate.side_effect = [
            {
                "is_valid": False,
                "issues": [
                    {"feature": "structures", "issue_type": "missing"}
                ],
            },
            {"is_valid": True, "issues": []},
        ]
        mock_validator_class.return_value = mock_validator

        mock_filter = AsyncMock()
        mock_filter.filter_for_features = AsyncMock(
            return_value={"structures": "Structures must be 50 feet"}
        )
        mock_filter_class.return_value = mock_filter

        mock_parser.reextract_focused_features.return_value = {
            "structures": {"value": 50, "units": "feet"}
        }

        graph = IterativeExtractionGraph(
            config=sample_config, llm_service=mock_llm_service
        )

        result = await graph.run(
            text="Sample ordinance text",
            schema=sample_schema,
            initial_extraction=sample_extraction,
            parser=mock_parser,
        )

        assert result["extraction"]["structures"]["value"] == 50
        assert result["metadata"]["iterations"] == 1
        assert result["metadata"]["final_valid"] is True


@pytest.mark.asyncio
async def test_run_max_iterations_reached(
    mock_llm_service, mock_parser, sample_config, sample_extraction, sample_schema
):
    """Test workflow when max iterations reached"""
    config_with_low_max = {**sample_config, "max_iterations": 1}

    with patch(
        "compass.extraction.iterative.ExtractionValidator"
    ) as mock_validator_class, patch(
        "compass.extraction.iterative.FocusedTextFilter"
    ) as mock_filter_class:
        mock_validator = AsyncMock()
        mock_validator.validate = AsyncMock(
            return_value={
                "is_valid": False,
                "issues": [
                    {"feature": "structures", "issue_type": "missing"}
                ],
            }
        )
        mock_validator_class.return_value = mock_validator

        mock_filter = AsyncMock()
        mock_filter.filter_for_features = AsyncMock(
            return_value={"structures": "Text"}
        )
        mock_filter_class.return_value = mock_filter

        graph = IterativeExtractionGraph(
            config=config_with_low_max, llm_service=mock_llm_service
        )

        result = await graph.run(
            text="Sample ordinance text",
            schema=sample_schema,
            initial_extraction=sample_extraction,
            parser=mock_parser,
        )

        assert result["metadata"]["iterations"] == 1


@pytest.mark.asyncio
async def test_run_with_parser_not_supporting_reextract(
    mock_llm_service, sample_config, sample_extraction, sample_schema
):
    """Test workflow when parser doesn't support focused re-extraction"""
    parser_without_method = MagicMock()

    with patch(
        "compass.extraction.iterative.ExtractionValidator"
    ) as mock_validator_class, patch(
        "compass.extraction.iterative.FocusedTextFilter"
    ) as mock_filter_class:
        mock_validator = AsyncMock()
        mock_validator.validate = AsyncMock(
            return_value={
                "is_valid": False,
                "issues": [
                    {"feature": "structures", "issue_type": "missing"}
                ],
            }
        )
        mock_validator_class.return_value = mock_validator

        mock_filter = AsyncMock()
        mock_filter.filter_for_features = AsyncMock(
            return_value={"structures": "Text"}
        )
        mock_filter_class.return_value = mock_filter

        graph = IterativeExtractionGraph(
            config=sample_config, llm_service=mock_llm_service
        )

        result = await graph.run(
            text="Sample ordinance text",
            schema=sample_schema,
            initial_extraction=sample_extraction,
            parser=parser_without_method,
        )

        assert "extraction" in result
        assert "metadata" in result


@pytest.mark.asyncio
async def test_run_with_error(
    mock_llm_service, mock_parser, sample_config, sample_extraction, sample_schema
):
    """Test workflow handles errors gracefully"""
    with patch(
        "compass.extraction.iterative.ExtractionValidator"
    ) as mock_validator_class:
        mock_validator = AsyncMock()
        mock_validator.validate = AsyncMock(side_effect=Exception("Test error"))
        mock_validator_class.return_value = mock_validator

        graph = IterativeExtractionGraph(
            config=sample_config, llm_service=mock_llm_service
        )

        result = await graph.run(
            text="Sample ordinance text",
            schema=sample_schema,
            initial_extraction=sample_extraction,
            parser=mock_parser,
        )

        assert result["extraction"] == sample_extraction
        assert "error" in result["metadata"]
        assert result["metadata"]["iterations"] == 0


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
