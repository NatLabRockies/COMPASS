"""COMPASS one-shot plugin configuration tests"""

from pathlib import Path
from warnings import catch_warnings, simplefilter

import pytest

from compass.exceptions import COMPASSPluginConfigurationError
from compass.plugin.utilities import normalize_website_keywords

from compass.warn import COMPASSPluginConfigurationWarning


def test_normalize_website_keywords_computes_tier_weights():
    """Website keyword tiers should have lexicographic priority"""

    keywords = normalize_website_keywords(
        [
            "pdf",
            "secs",
            "solar",
            "zoning",
            "ordinance",
            "renewable energy",
            "planning",
            "plan",
            "government",
            ["code", "area"],
            "land development",
            ["land", "environment", "energy", "renewable"],
            ["municipal", "department"],
        ]
    )

    assert keywords["pdf"] == 92160
    assert keywords["secs"] == 46080
    assert keywords["solar"] == 23040
    assert keywords["renewable energy"] == 1440
    assert keywords["renewable%20energy"] == 1440
    assert keywords["renewable+energy"] == 1440
    assert keywords["municipal"] == 1
    assert keywords["department"] == 1
    assert keywords["ordinance"] > sum(
        keywords[keyword]
        for keyword in (
            "renewable energy",
            "renewable%20energy",
            "renewable+energy",
            "planning",
            "plan",
            "government",
            "code",
            "area",
            "land development",
            "land%20development",
            "land+development",
            "land",
            "environment",
            "energy",
            "renewable",
            "municipal",
            "department",
        )
    )


def test_normalize_website_keywords_matches_oil_gas_tiers():
    """Oil and gas keyword tiers should retain their established scores"""

    keywords = normalize_website_keywords(
        [
            ["pdf"],
            [
                "oil and gas",
                "oil & gas",
                "gas and oil",
                "oil-and-gas",
                "oil_gas",
                "oil well",
                "gas well",
                "oil-well",
                "gas-well",
            ],
            [
                "wellhead",
                "well pad",
                "well-pad",
                "drilling permit",
                "drilling ordinance",
                "hydraulic fracturing",
                "fracking",
                "drilling",
            ],
            ["planning"],
            ["plan"],
            ["government"],
            ["code", "area"],
            [
                "code of ordinances",
                "code_of_ordinances",
                "land use code",
                "land_use_code",
                "land development",
            ],
            ["ordinance", "zoning"],
            ["land", "environment"],
            ["municipal", "department"],
        ]
    )

    assert keywords["pdf"] == 2643840
    assert keywords["oil and gas"] == 132192
    assert keywords["well pad"] == 7776
    assert keywords["planning"] == 3888
    assert keywords["code of ordinances"] == 27
    assert keywords["ordinance"] == 9


def test_normalize_website_keywords_accepts_flat_mappings():
    """Legacy website keyword score mappings should remain supported"""

    with pytest.warns(COMPASSPluginConfigurationWarning):
        keywords = normalize_website_keywords({"wind energy": 10, "wecs": 5})

    assert keywords == {
        "wind energy": 10,
        "wecs": 5,
        "wind%20energy": 10,
        "wind+energy": 10,
    }


def test_normalize_website_keywords_warns_for_missing_ordinance_terms():
    """Missing ordinance-oriented keywords should explain their value"""

    with pytest.warns(
        COMPASSPluginConfigurationWarning,
        match="help push link prioritization toward ordinance documents",
    ):
        normalize_website_keywords([["pdf"], ["wind"]])


def test_normalize_website_keywords_allows_zero_score_opt_outs():
    """Explicit zero-score keywords should silence the guidance warning"""

    with catch_warnings(record=True) as captured:
        simplefilter("always")
        normalize_website_keywords(
            {
                "planning": 0,
                "plan": 0,
                "government": 0,
                "zoning": 0,
                "land": 0,
                "municipal": 0,
                "department": 0,
            }
        )

    assert not captured


@pytest.mark.parametrize(
    ("tiers", "match"),
    [
        ([], "at least one keyword tier"),
        ([[]], "tier 1 must be a non-empty list or string"),
        ([1], "tier 1 must be a non-empty list or string"),
        ([["wind", "wind"]], "duplicate keywords"),
        ([["wind energy", "wind%20energy"]], "URL variants"),
        ([["wind"], ["wind"]], "duplicate keywords"),
    ],
)
def test_normalize_website_keywords_rejects_invalid_tiers(tiers, match):
    """Website keyword tiers should have unique, non-empty terms"""

    with pytest.raises(COMPASSPluginConfigurationError, match=match):
        normalize_website_keywords(tiers)


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
