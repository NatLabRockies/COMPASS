"""Test iterative validation module"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from compass.extraction.iterative_validation import ExtractionValidator


@pytest.fixture
def mock_llm_caller():
    """Mock LLM caller for validation tests"""
    with patch(
        "compass.extraction.iterative_validation.SchemaOutputLLMCaller"
    ) as mock_caller_class:
        mock_caller = AsyncMock()
        mock_caller_class.return_value = mock_caller
        yield mock_caller


@pytest.fixture
def sample_schema():
    """Sample schema for testing"""
    return {
        "type": "object",
        "properties": {
            "property_line": {
                "type": "object",
                "description": "Distance to property line",
                "properties": {
                    "value": {"type": "number"},
                    "units": {"type": "string"},
                },
            },
            "structures": {
                "type": "object",
                "description": "Distance to structures",
                "properties": {
                    "value": {"type": "number"},
                    "units": {"type": "string"},
                },
            },
        },
    }


@pytest.fixture
def sample_extraction():
    """Sample extraction result"""
    return {
        "property_line": {"value": 100, "units": "feet"},
        "structures": {"value": None, "units": None},
    }


def test_validator_initialization(mock_llm_caller):
    """Test validator initializes correctly"""
    validator = ExtractionValidator(
        llm_caller=MagicMock(), strictness="moderate"
    )
    assert validator is not None


def test_validator_strictness_validation():
    """Test strictness parameter validation"""
    with pytest.raises(ValueError, match="Invalid strictness"):
        ExtractionValidator(llm_caller=MagicMock(), strictness="invalid")


@pytest.mark.asyncio
async def test_check_completeness_all_present(mock_llm_caller, sample_schema):
    """Test completeness check when all features present"""
    mock_llm_caller.call.return_value = {
        "missing_features": [],
        "reasoning": "All features present",
    }

    validator = ExtractionValidator(
        llm_caller=mock_llm_caller, strictness="moderate"
    )

    extraction = {
        "property_line": {"value": 100, "units": "feet"},
        "structures": {"value": 50, "units": "feet"},
    }

    result = await validator._check_completeness(
        extraction, sample_schema, "Sample ordinance text"
    )

    assert result["missing_features"] == []
    assert result["complete"] is True


@pytest.mark.asyncio
async def test_check_completeness_missing_features(
    mock_llm_caller, sample_schema
):
    """Test completeness check detects missing features"""
    mock_llm_caller.call.return_value = {
        "missing_features": [
            {
                "feature_id": "structures",
                "confidence": "high",
                "reasoning": "Value is None but required",
            }
        ],
        "reasoning": "One feature missing",
    }

    validator = ExtractionValidator(
        llm_caller=mock_llm_caller, strictness="moderate"
    )

    extraction = {
        "property_line": {"value": 100, "units": "feet"},
        "structures": {"value": None, "units": None},
    }

    result = await validator._check_completeness(
        extraction, sample_schema, "Sample ordinance text"
    )

    assert len(result["missing_features"]) == 1
    assert result["missing_features"][0]["feature_id"] == "structures"
    assert result["complete"] is False


@pytest.mark.asyncio
async def test_check_correctness_all_correct(mock_llm_caller, sample_schema):
    """Test correctness check when all features correct"""
    mock_llm_caller.call.return_value = {
        "incorrect_features": [],
        "reasoning": "All features correct",
    }

    validator = ExtractionValidator(
        llm_caller=mock_llm_caller, strictness="moderate"
    )

    extraction = {
        "property_line": {"value": 100, "units": "feet"},
        "structures": {"value": 50, "units": "feet"},
    }

    result = await validator._check_correctness(
        extraction, sample_schema, "Sample ordinance text"
    )

    assert result["incorrect_features"] == []
    assert result["correct"] is True


@pytest.mark.asyncio
async def test_check_correctness_incorrect_features(
    mock_llm_caller, sample_schema
):
    """Test correctness check detects incorrect features"""
    mock_llm_caller.call.return_value = {
        "incorrect_features": [
            {
                "feature_id": "property_line",
                "confidence": "high",
                "issue": "Value should be 200 not 100",
            }
        ],
        "reasoning": "One feature incorrect",
    }

    validator = ExtractionValidator(
        llm_caller=mock_llm_caller, strictness="moderate"
    )

    extraction = {
        "property_line": {"value": 100, "units": "feet"},
        "structures": {"value": 50, "units": "feet"},
    }

    result = await validator._check_correctness(
        extraction, sample_schema, "Sample ordinance text"
    )

    assert len(result["incorrect_features"]) == 1
    assert result["incorrect_features"][0]["feature_id"] == "property_line"
    assert result["correct"] is False


@pytest.mark.asyncio
async def test_validate_extraction_pass(mock_llm_caller, sample_schema):
    """Test full validation passes when extraction is complete and correct"""
    mock_llm_caller.call.side_effect = [
        {"missing_features": [], "reasoning": "Complete"},
        {"incorrect_features": [], "reasoning": "Correct"},
    ]

    validator = ExtractionValidator(
        llm_caller=mock_llm_caller, strictness="moderate"
    )

    extraction = {
        "property_line": {"value": 100, "units": "feet"},
        "structures": {"value": 50, "units": "feet"},
    }

    result = await validator.validate_extraction(
        extraction, sample_schema, "Sample ordinance text"
    )

    assert result["is_valid"] is True
    assert result["issues"] == []


@pytest.mark.asyncio
async def test_validate_extraction_fail(mock_llm_caller, sample_schema):
    """Test full validation fails when issues detected"""
    mock_llm_caller.call.side_effect = [
        {
            "missing_features": [
                {
                    "feature_id": "structures",
                    "confidence": "high",
                    "reasoning": "Missing",
                }
            ],
            "reasoning": "Incomplete",
        },
        {"incorrect_features": [], "reasoning": "Correct"},
    ]

    validator = ExtractionValidator(
        llm_caller=mock_llm_caller, strictness="moderate"
    )

    extraction = {
        "property_line": {"value": 100, "units": "feet"},
        "structures": {"value": None, "units": None},
    }

    result = await validator.validate_extraction(
        extraction, sample_schema, "Sample ordinance text"
    )

    assert result["is_valid"] is False
    assert len(result["issues"]) == 1
    assert result["issues"][0]["feature"] == "structures"
    assert result["issues"][0]["issue_type"] == "missing"


@pytest.mark.asyncio
async def test_strictness_lenient_filters_low_confidence(
    mock_llm_caller, sample_schema
):
    """Test lenient strictness filters out low confidence issues"""
    mock_llm_caller.call.side_effect = [
        {
            "missing_features": [
                {
                    "feature_id": "structures",
                    "confidence": "low",
                    "reasoning": "Maybe missing",
                }
            ],
            "reasoning": "Possibly incomplete",
        },
        {"incorrect_features": [], "reasoning": "Correct"},
    ]

    validator = ExtractionValidator(
        llm_caller=mock_llm_caller, strictness="lenient"
    )

    extraction = {
        "property_line": {"value": 100, "units": "feet"},
        "structures": {"value": None, "units": None},
    }

    result = await validator.validate_extraction(
        extraction, sample_schema, "Sample ordinance text"
    )

    assert result["is_valid"] is True
    assert result["issues"] == []


@pytest.mark.asyncio
async def test_strictness_strict_includes_all_issues(
    mock_llm_caller, sample_schema
):
    """Test strict strictness includes all confidence levels"""
    mock_llm_caller.call.side_effect = [
        {
            "missing_features": [
                {
                    "feature_id": "structures",
                    "confidence": "low",
                    "reasoning": "Maybe missing",
                }
            ],
            "reasoning": "Possibly incomplete",
        },
        {"incorrect_features": [], "reasoning": "Correct"},
    ]

    validator = ExtractionValidator(
        llm_caller=mock_llm_caller, strictness="strict"
    )

    extraction = {
        "property_line": {"value": 100, "units": "feet"},
        "structures": {"value": None, "units": None},
    }

    result = await validator.validate_extraction(
        extraction, sample_schema, "Sample ordinance text"
    )

    assert result["is_valid"] is False
    assert len(result["issues"]) == 1


@pytest.mark.asyncio
async def test_get_problematic_features(mock_llm_caller, sample_schema):
    """Test getting list of problematic features"""
    mock_llm_caller.call.side_effect = [
        {
            "missing_features": [
                {
                    "feature_id": "structures",
                    "confidence": "high",
                    "reasoning": "Missing",
                }
            ],
            "reasoning": "Incomplete",
        },
        {
            "incorrect_features": [
                {
                    "feature_id": "property_line",
                    "confidence": "high",
                    "issue": "Wrong value",
                }
            ],
            "reasoning": "Incorrect",
        },
    ]

    validator = ExtractionValidator(
        llm_caller=mock_llm_caller, strictness="moderate"
    )

    extraction = {
        "property_line": {"value": 100, "units": "feet"},
        "structures": {"value": None, "units": None},
    }

    result = await validator.validate_extraction(
        extraction, sample_schema, "Sample ordinance text"
    )

    features = validator.get_problematic_features(result)
    assert set(features) == {"structures", "property_line"}


def test_get_problematic_features_empty():
    """Test getting features from empty validation result"""
    validator = ExtractionValidator(
        llm_caller=MagicMock(), strictness="moderate"
    )
    result = {"is_valid": True, "issues": []}
    features = validator.get_problematic_features(result)
    assert features == []


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
