"""COMPASS one-shot plugin configuration tests"""

import json
from pathlib import Path
from warnings import catch_warnings, simplefilter

import pytest

from compass.exceptions import COMPASSPluginConfigurationError
from compass.plugin.one_shot.base import (
    _normalize_website_keywords,
    _out_cols_from_config,
    create_schema_based_one_shot_extraction_plugin,
)
from compass.warn import COMPASSPluginConfigurationWarning


def test_out_cols_from_config_uses_schema_output_properties():
    """Test schema output fields become output columns"""

    config = {
        "schema": {
            "properties": {
                "outputs": {
                    "items": {
                        "required": [
                            "feature",
                            "value",
                            "units",
                            "location",
                            "summary",
                            "section",
                            "source",
                        ],
                        "properties": {
                            "feature": {},
                            "value": {},
                            "units": {},
                            "location": {},
                            "summary": {},
                            "section": {},
                            "source": {},
                        },
                    }
                }
            }
        }
    }

    cols = _out_cols_from_config(config)

    assert [col.name for col in cols] == [
        "county",
        "state",
        "subdivision",
        "jurisdiction_type",
        "FIPS",
        "feature",
        "value",
        "units",
        "location",
        "summary",
        "section",
        "year",
        "source",
        "quantitative",
    ]
    assert (
        next(col for col in cols if col.name == "value").include_in_qual_output
        is False
    )
    assert (
        next(col for col in cols if col.name == "units").include_in_qual_output
        is False
    )
    assert (
        next(
            col for col in cols if col.name == "location"
        ).include_in_qual_output
        is True
    )


def test_normalize_website_keywords_computes_tier_weights():
    """Website keyword tiers should have lexicographic priority"""

    keywords = _normalize_website_keywords(
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

    keywords = _normalize_website_keywords(
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
        keywords = _normalize_website_keywords({"wind energy": 10, "wecs": 5})

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
        _normalize_website_keywords([["pdf"], ["wind"]])


def test_normalize_website_keywords_allows_zero_score_opt_outs():
    """Explicit zero-score keywords should silence the guidance warning"""

    with catch_warnings(record=True) as captured:
        simplefilter("always")
        _normalize_website_keywords(
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
        _normalize_website_keywords(tiers)


@pytest.mark.asyncio
async def test_one_shot_plugin_loads_tiered_website_keywords(tmp_path):
    """One-shot plugin configs should expose computed keyword scores"""

    schema_fp = tmp_path / "schema.json"
    schema_fp.write_text(
        json.dumps(
            {
                "properties": {
                    "outputs": {
                        "items": {
                            "required": ["feature"],
                            "properties": {"feature": {}},
                        }
                    }
                }
            }
        )
    )
    config_fp = tmp_path / "plugin.yaml"
    config_fp.write_text(
        f"schema: {schema_fp}\n"
        "website_keywords:\n"
        "  - pdf\n"
        "  - wind energy\n"
        "  - ordinance\n"
    )

    plugin_class = create_schema_based_one_shot_extraction_plugin(
        config_fp, tech=f"tiered-keywords-{tmp_path.name}"
    )
    with pytest.warns(COMPASSPluginConfigurationWarning):
        keywords = await plugin_class(None, {}).get_website_keywords()

    assert keywords == {
        "pdf": 8,
        "wind energy": 2,
        "wind%20energy": 2,
        "wind+energy": 2,
        "ordinance": 1,
    }


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
