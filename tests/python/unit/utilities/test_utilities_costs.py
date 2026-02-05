"""Tests for COMPASS cost computation utilities"""

from pathlib import Path

import pytest

from compass.utilities.costs import (
    LLM_COST_REGISTRY,
    compute_cost_from_totals,
    compute_total_cost_from_usage,
    cost_for_model,
)


@pytest.mark.parametrize(
    "model_name,prompt_tokens,completion_tokens,expected",
    [
        ("gpt-4o", 1_000_000, 1_000_000, 12.5),
        ("gpt-4o-mini", 1_000_000, 1_000_000, 0.75),
        ("o1", 500_000, 500_000, 37.5),
        ("gpt-5-nano", 2_000_000, 1_000_000, 0.5),
        ("unknown-model", 1_000_000, 1_000_000, 0.0),
        ("gpt-4o", 0, 0, 0.0),
        ("gpt-4o", 100_000, 0, 0.25),
        ("gpt-4o", 0, 100_000, 1.0),
        ("gpt-4o", 2_500_000, 3_000_000, 36.25),
        ("compassop-gpt-4.1-nano", 1_000_000, 500_000, 0.3),
        ("wetosa-gpt-5-mini", 500_000, 500_000, 1.125),
    ],
)
def test_cost_for_model_known_models(
    model_name, prompt_tokens, completion_tokens, expected
):
    """Test `cost_for_model` with various known models and token counts"""
    result = cost_for_model(model_name, prompt_tokens, completion_tokens)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "model_name,prompt_tokens,completion_tokens",
    [
        ("", 1_000_000, 1_000_000),
        ("GPT-4O", 1_000_000, 1_000_000),
        ("gpt-4o-MINI", 1_000_000, 1_000_000),
        ("gpt4o", 1_000_000, 1_000_000),
    ],
)
def test_cost_for_model_case_sensitivity_and_unknown(
    model_name, prompt_tokens, completion_tokens
):
    """Test `cost_for_model` returns zero for bad inputs"""
    result = cost_for_model(model_name, prompt_tokens, completion_tokens)
    assert result == 0.0


def test_cost_for_model_with_embedding_model():
    """Test `cost_for_model` with embedding-only model"""
    result = cost_for_model("text-embedding-ada-002", 1_000_000, 0)
    assert result == pytest.approx(0.10)


def test_cost_for_model_with_large_token_counts():
    """Test `cost_for_model` handles very large token counts accurately"""
    result = cost_for_model("gpt-4o", 100_000_000, 50_000_000)
    assert result == pytest.approx(750.0)


@pytest.mark.parametrize(
    "totals,expected",
    [
        (
            {
                "gpt-4o": {
                    "prompt_tokens": 1_000_000,
                    "response_tokens": 500_000,
                }
            },
            7.5,
        ),
        (
            {
                "gpt-4o": {
                    "prompt_tokens": 1_000_000,
                    "response_tokens": 500_000,
                },
                "gpt-4o-mini": {
                    "prompt_tokens": 2_000_000,
                    "response_tokens": 1_000_000,
                },
            },
            8.4,
        ),
        ({}, 0.0),
        (
            {"gpt-4o": {}},
            0.0,
        ),
        (
            {"gpt-4o": {"prompt_tokens": 1_000_000}},
            2.5,
        ),
        (
            {"gpt-4o": {"response_tokens": 1_000_000}},
            10.0,
        ),
        (
            {
                "gpt-4o": {
                    "prompt_tokens": 500_000,
                    "response_tokens": 200_000,
                },
                "unknown-model": {
                    "prompt_tokens": 1_000_000,
                    "response_tokens": 1_000_000,
                },
            },
            3.25,
        ),
        (
            {
                "o1": {"prompt_tokens": 1_000_000, "response_tokens": 500_000},
                "gpt-5-nano": {
                    "prompt_tokens": 2_000_000,
                    "response_tokens": 1_000_000,
                },
                "gpt-4.1-mini": {
                    "prompt_tokens": 500_000,
                    "response_tokens": 500_000,
                },
            },
            46.5,
        ),
    ],
)
def test_compute_cost_from_totals(totals, expected):
    """Test `compute_cost_from_totals` with various total configurations"""
    result = compute_cost_from_totals(totals)
    assert result == pytest.approx(expected)


def test_compute_cost_from_totals_with_extra_keys():
    """Test `compute_cost_from_totals` ignores extra keys in usage dict"""
    totals = {
        "gpt-4o": {
            "prompt_tokens": 1_000_000,
            "response_tokens": 500_000,
            "extra_key": "ignored",
            "another_key": 999,
        }
    }
    result = compute_cost_from_totals(totals)
    assert result == pytest.approx(7.5)


@pytest.mark.parametrize(
    "tracked_usage,expected",
    [
        (
            {
                "location1": {
                    "tracker_totals": {
                        "gpt-4o": {
                            "prompt_tokens": 1_000_000,
                            "response_tokens": 500_000,
                        }
                    }
                }
            },
            7.5,
        ),
        (
            {
                "location1": {
                    "tracker_totals": {
                        "gpt-4o": {
                            "prompt_tokens": 1_000_000,
                            "response_tokens": 500_000,
                        }
                    }
                },
                "location2": {
                    "tracker_totals": {
                        "gpt-4o-mini": {
                            "prompt_tokens": 2_000_000,
                            "response_tokens": 1_000_000,
                        }
                    }
                },
            },
            8.4,
        ),
        ({}, 0.0),
        (
            {"location1": {}},
            0.0,
        ),
        (
            {"location1": {"tracker_totals": {}}},
            0.0,
        ),
        (
            {
                "location1": {
                    "tracker_totals": {
                        "gpt-4o": {
                            "prompt_tokens": 500_000,
                            "response_tokens": 200_000,
                        }
                    }
                },
                "location2": {
                    "tracker_totals": {
                        "unknown-model": {
                            "prompt_tokens": 1_000_000,
                            "response_tokens": 1_000_000,
                        }
                    }
                },
            },
            3.25,
        ),
        (
            {
                "new_york_county": {
                    "tracker_totals": {
                        "gpt-4o": {
                            "prompt_tokens": 800_000,
                            "response_tokens": 400_000,
                        },
                        "gpt-4o-mini": {
                            "prompt_tokens": 1_500_000,
                            "response_tokens": 750_000,
                        },
                    }
                },
                "california_county": {
                    "tracker_totals": {
                        "o1": {
                            "prompt_tokens": 500_000,
                            "response_tokens": 250_000,
                        },
                    }
                },
                "texas_county": {
                    "tracker_totals": {
                        "gpt-5-nano": {
                            "prompt_tokens": 3_000_000,
                            "response_tokens": 2_000_000,
                        },
                    }
                },
            },
            30.125,
        ),
    ],
)
def test_compute_total_cost_from_usage(tracked_usage, expected):
    """Test `compute_total_cost_from_usage` with various usage configs"""
    result = compute_total_cost_from_usage(tracked_usage)
    assert result == pytest.approx(expected)


def test_compute_total_cost_from_usage_with_extra_keys():
    """Test `compute_total_cost_from_usage` ignores extra keys in usage dict"""
    tracked_usage = {
        "location1": {
            "tracker_totals": {
                "gpt-4o": {
                    "prompt_tokens": 1_000_000,
                    "response_tokens": 500_000,
                }
            },
            "extra_field": "ignored",
            "timestamp": "2026-01-01",
        }
    }
    result = compute_total_cost_from_usage(tracked_usage)
    assert result == pytest.approx(7.5)


def test_integration_single_jurisdiction_workflow():
    """Test complete workflow from model costs to total tracked usage"""
    prompt_tokens = 1_000_000
    completion_tokens = 500_000

    model_cost = cost_for_model("gpt-4o", prompt_tokens, completion_tokens)
    assert model_cost == pytest.approx(7.5)

    totals = {
        "gpt-4o": {
            "prompt_tokens": prompt_tokens,
            "response_tokens": completion_tokens,
        }
    }
    totals_cost = compute_cost_from_totals(totals)
    assert totals_cost == pytest.approx(7.5)
    assert totals_cost == pytest.approx(model_cost)

    tracked_usage = {"jurisdiction1": {"tracker_totals": totals}}
    total_cost = compute_total_cost_from_usage(tracked_usage)
    assert total_cost == pytest.approx(7.5)
    assert total_cost == pytest.approx(totals_cost)


def test_integration_multi_jurisdiction_multi_model_workflow():
    """Test complete workflow with multiple jurisdictions and models"""
    jurisdiction_configs = [
        ("california", "gpt-4o", 1_000_000, 500_000),
        ("texas", "gpt-4o-mini", 2_000_000, 1_000_000),
        ("new_york", "o1", 500_000, 250_000),
    ]

    expected_individual_costs = []
    tracked_usage = {}

    for jurisdiction, model, prompt, completion in jurisdiction_configs:
        individual_cost = cost_for_model(model, prompt, completion)
        expected_individual_costs.append(individual_cost)

        totals = {
            model: {"prompt_tokens": prompt, "response_tokens": completion}
        }
        totals_cost = compute_cost_from_totals(totals)
        assert totals_cost == pytest.approx(individual_cost)

        tracked_usage[jurisdiction] = {"tracker_totals": totals}

    total_cost = compute_total_cost_from_usage(tracked_usage)
    expected_total = sum(expected_individual_costs)
    assert total_cost == pytest.approx(expected_total)
    assert total_cost == pytest.approx(30.9)


def test_integration_mixed_known_unknown_models():
    """Test integration with mix of known and unknown models"""
    totals = {
        "gpt-4o": {"prompt_tokens": 500_000, "response_tokens": 200_000},
        "unknown-model-1": {
            "prompt_tokens": 1_000_000,
            "response_tokens": 1_000_000,
        },
        "gpt-4o-mini": {
            "prompt_tokens": 1_000_000,
            "response_tokens": 500_000,
        },
        "unknown-model-2": {
            "prompt_tokens": 500_000,
            "response_tokens": 500_000,
        },
    }

    totals_cost = compute_cost_from_totals(totals)

    tracked_usage = {"jurisdiction": {"tracker_totals": totals}}
    total_cost = compute_total_cost_from_usage(tracked_usage)

    assert totals_cost == pytest.approx(total_cost)
    assert total_cost == pytest.approx(3.7)


def test_llm_cost_registry_structure():
    """Test LLM_COST_REGISTRY has expected structure"""
    assert isinstance(LLM_COST_REGISTRY, dict)
    assert len(LLM_COST_REGISTRY) > 0

    for model_name, costs in LLM_COST_REGISTRY.items():
        assert isinstance(model_name, str)
        assert len(model_name) > 0
        assert isinstance(costs, dict)
        assert "prompt" in costs
        assert isinstance(costs["prompt"], (int, float))
        assert costs["prompt"] > 0


def test_llm_cost_registry_response_costs():
    """Test models with response costs have valid values"""
    models_with_response = [
        model
        for model, costs in LLM_COST_REGISTRY.items()
        if "response" in costs
    ]

    assert len(models_with_response) > 0

    for model in models_with_response:
        response_cost = LLM_COST_REGISTRY[model]["response"]
        assert isinstance(response_cost, (int, float))
        assert response_cost > 0


def test_llm_cost_registry_embedding_models():
    """Test embedding models have prompt cost but may lack response cost"""
    embedding_model = "text-embedding-ada-002"
    assert embedding_model in LLM_COST_REGISTRY
    assert "prompt" in LLM_COST_REGISTRY[embedding_model]
    assert "response" not in LLM_COST_REGISTRY[embedding_model]


def test_llm_cost_registry_model_name_patterns():
    """Test registry contains expected model name patterns"""
    model_names = list(LLM_COST_REGISTRY.keys())

    assert any("gpt-4o" in name for name in model_names)
    assert any("gpt-5" in name for name in model_names)
    assert any("compassop" in name for name in model_names)
    assert any("wetosa" in name for name in model_names)


def test_llm_cost_registry_response_higher_than_prompt():
    """Test response costs are typically higher than prompt costs"""
    models_with_both = [
        model
        for model, costs in LLM_COST_REGISTRY.items()
        if "response" in costs and "prompt" in costs
    ]

    higher_response_count = sum(
        1
        for model in models_with_both
        if LLM_COST_REGISTRY[model]["response"]
        > LLM_COST_REGISTRY[model]["prompt"]
    )

    assert higher_response_count > len(models_with_both) * 0.8


def test_llm_cost_registry_no_negative_costs():
    """Test registry contains no negative cost values"""
    for model_name, costs in LLM_COST_REGISTRY.items():
        for cost_type, cost_value in costs.items():
            assert cost_value >= 0, (
                f"Negative cost for {model_name}.{cost_type}"
            )


def test_cost_for_model_with_negative_tokens():
    """Test `cost_for_model` handles negative token counts as zero"""
    result = cost_for_model("gpt-4o", -1_000_000, -500_000)
    assert result == pytest.approx(-7.5)


def test_compute_cost_from_totals_with_negative_tokens():
    """Test `compute_cost_from_totals` with negative token values"""
    totals = {
        "gpt-4o": {"prompt_tokens": -1_000_000, "response_tokens": 500_000}
    }
    result = compute_cost_from_totals(totals)
    assert result == pytest.approx(2.5)


def test_cost_calculation_precision():
    """Test cost calculations maintain precision with small token counts"""
    result = cost_for_model("gpt-4o", 1, 1)
    expected = (1 / 1e6 * 2.5) + (1 / 1e6 * 10)
    assert result == pytest.approx(expected)
    assert result == pytest.approx(0.0000125)


def test_compute_total_cost_from_usage_deeply_nested():
    """Test `compute_total_cost_from_usage` with realistic nested structure"""
    tracked_usage = {
        "jurisdiction_1": {
            "tracker_totals": {
                "gpt-4o": {
                    "prompt_tokens": 500_000,
                    "response_tokens": 250_000,
                },
                "gpt-4o-mini": {
                    "prompt_tokens": 1_000_000,
                    "response_tokens": 500_000,
                },
            },
            "metadata": {"runtime": 120.5},
        },
        "jurisdiction_2": {
            "tracker_totals": {
                "gpt-5-nano": {
                    "prompt_tokens": 2_000_000,
                    "response_tokens": 1_000_000,
                },
            },
            "metadata": {"runtime": 95.3},
        },
    }

    result = compute_total_cost_from_usage(tracked_usage)
    expected = (
        (500_000 / 1e6 * 2.5 + 250_000 / 1e6 * 10)
        + (1_000_000 / 1e6 * 0.15 + 500_000 / 1e6 * 0.6)
        + (2_000_000 / 1e6 * 0.05 + 1_000_000 / 1e6 * 0.4)
    )
    assert result == pytest.approx(expected)
    assert result == pytest.approx(4.7)


def test_compute_jurisdiction_cost_uses_registry():
    """Ensure model costs are computed using registry values"""

    tracker = {
        "jurisdiction_1": {
            "tracker_totals": {
                "gpt-4o": {
                    "prompt_tokens": 1_000_000,
                    "response_tokens": 1_000_000,
                }
            }
        }
    }
    assert compute_total_cost_from_usage(tracker) == pytest.approx(12.5)

    tracker_unknown = {
        "jurisdiction_1": {
            "tracker_totals": {"unknown": {"prompt_tokens": 1_000_000}}
        }
    }
    assert compute_total_cost_from_usage(tracker_unknown) == 0


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
