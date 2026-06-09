"""COMPASS one-shot plugin tests"""

from pathlib import Path

import pytest

from compass.exceptions import COMPASSPluginConfigurationError
from compass.plugin.one_shot.base import _normalize_heuristic_keywords
from compass.warn import COMPASSPluginConfigurationWarning


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


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
