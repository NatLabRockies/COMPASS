"""COMPASS one-shot plugin configuration tests"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd
import pyjson5

from compass import COMPASSSolarOneShotExtractor, COMPASSWindOneShotExtractor
from compass.extraction.features import SetbackFeatures
from compass.extraction.solar.ordinance import (
    SolarOrdinanceTextCollector,
    SolarOrdinanceTextExtractor,
)
from compass.extraction.solar.parse import (
    EXTRA_NUMERICAL_RESTRICTIONS,
    EXTRA_QUALITATIVE_RESTRICTIONS,
    StructuredSolarPermittedUseDistrictsParser,
)
from compass.extraction.wind.ordinance import (
    WindOrdinanceTextCollector,
    WindOrdinanceTextExtractor,
)
from compass.extraction.wind.parse import (
    EXTRA_NUMERICAL_RESTRICTIONS as WIND_NUMERICAL_RESTRICTIONS,
    EXTRA_QUALITATIVE_RESTRICTIONS as WIND_QUALITATIVE_RESTRICTIONS,
    StructuredWindPermittedUseDistrictsParser,
)
from compass.plugin.one_shot.base import (
    _out_cols_from_config,
    create_schema_based_one_shot_extraction_plugin,
)
from compass.warn import COMPASSPluginConfigurationWarning
from compass.plugin.one_shot.components import SchemaOrdinanceParser
from compass.utilities.finalize import doc_infos_to_db, save_db
from compass.utilities.jurisdictions import Jurisdiction


@pytest.mark.parametrize(
    "schema_path",
    [
        "compass/extraction/geothermal_electricity/geothermal_schema.json",
        "compass/extraction/ghp/geothermal_heat_pump_schema.json5",
        # "compass/extraction/natural_gas/natural_gas_schema.json",
        # (
        #     "compass/extraction/natural_gas_pipelines/"
        #     "natural_gas_pipelines_schema.json"
        # ),
        "compass/extraction/oil_gas_wells/oil_and_gas_wells_schema.json",
        "compass/extraction/rmp/rmp_schema.json",
        "compass/extraction/solar_os/solar_schema.json5",
        # "compass/extraction/transmission/transmission_schema.json5",
        "compass/extraction/wind_os/wind_schema.json5",
        "examples/water_rights_demo/one-shot/water_rights_schema.json5",
    ],
)
def test_schema_summary_and_evidence_survive_export(
    schema_path, test_data_dir, tmp_path
):
    """Keep summary text independent of evidence through parsing and export"""
    schema = pyjson5.decode(
        (test_data_dir.parent.parent / schema_path).read_text()
    )
    items = schema["properties"]["outputs"]["items"]
    fields = {"summary", "ordinance_text", "explanation"}
    assert fields <= set(items["required"])
    columns = _out_cols_from_config({"schema": schema})
    qualitative = set(schema.get("$qualitative_features", []))
    features = items["properties"]["feature"]["enum"]
    selected = [next(name for name in features if name not in qualitative)]
    if qualitative:
        selected.append(next(name for name in features if name in qualitative))

    rows = []
    for feature in selected:
        row = dict.fromkeys(items["properties"])
        row.update(
            feature=feature,
            value=None if feature in qualitative else 100,
            units=None if feature in qualitative else "feet",
            summary='Summary: "Existing wording."\nJustification: Scope.',
            ordinance_text="The complete source passage, kept separately.",
            explanation="The passage supports this feature.",
            source="https://example.com/ordinance.pdf",
        )
        rows.append(row)

    parser = SimpleNamespace(
        SCHEMA=schema,
        QUALITATIVE_FEATURES=qualitative,
        POSSIBLE_OUT_COLS=columns,
    )
    parsed = SchemaOrdinanceParser._to_dataframe(parser, rows)
    shard = tmp_path / "extraction.csv"
    parsed.to_csv(shard, index=False)
    db, count = doc_infos_to_db(
        [
            {
                "ord_db_fp": shard,
                "jurisdiction": Jurisdiction("county", "Colorado", "Adams"),
            }
        ],
        columns,
    )
    save_db(db, tmp_path, columns)

    saved = pd.read_csv(
        tmp_path / "ordinances.csv", keep_default_na=False
    ).set_index("feature")
    assert count == 1
    assert set(saved.index) == set(selected)
    for row in rows:
        for field in fields:
            assert saved.loc[row["feature"], field] == row[field]


@pytest.mark.parametrize(
    "component, legacy_class",
    [
        ("TEXT_COLLECTORS", SolarOrdinanceTextCollector),
        ("TEXT_EXTRACTORS", SolarOrdinanceTextExtractor),
    ],
)
def test_solar_one_shot_prompts_match_legacy_solar(component, legacy_class):
    """Keep authored one-shot prompts aligned with legacy solar wording"""
    components = getattr(COMPASSSolarOneShotExtractor, component)
    assert len(components) == 1
    actual = components[0].PROMPTS
    expected = legacy_class.PROMPTS
    assert len(actual) == len(expected)
    for actual_prompt, expected_prompt in zip(actual, expected, strict=True):
        assert actual_prompt.keys() == expected_prompt.keys()
        for key, value in expected_prompt.items():
            assert " ".join(actual_prompt[key].split()) == " ".join(
                value.split()
            )


def test_solar_one_shot_retains_legacy_features_and_evidenced_examples():
    """Cover all solar features without introducing placeholder examples"""
    schema = COMPASSSolarOneShotExtractor.SCHEMA
    items = schema["properties"]["outputs"]["items"]
    features = items["properties"]["feature"]["enum"]
    expected = set(EXTRA_NUMERICAL_RESTRICTIONS)
    expected.update(EXTRA_QUALITATIVE_RESTRICTIONS)
    expected.update(
        entry["feature_id"]
        for entry in StructuredSolarPermittedUseDistrictsParser._USE_TYPES
    )
    for entry in SetbackFeatures():
        feature = entry["feature_id"]
        if feature in {"structures", "property line"}:
            expected.update(
                f"{feature} ({owner})"
                for owner in ("participating", "non-participating")
            )
        else:
            expected.add(feature)

    assert len(features) == len(set(features)) == 31
    assert set(features) == expected
    definitions = {
        feature: definition
        for family in schema["$definitions"].values()
        for feature, definition in family["properties"].items()
    }
    assert set(definitions) == expected
    for definition in definitions.values():
        for rule in ("VALUE:", "UNITS:", "IGNORE:"):
            assert rule in definition["description"]

    for case in schema["$examples"]:
        rows = case["outputs"]
        assert len(rows) == len({row["feature"] for row in rows})
        for row in rows:
            assert set(row) == set(items["required"])
            assert row["feature"] in expected
            for field in ("summary", "ordinance_text", "explanation"):
                assert isinstance(row[field], str) and row[field].strip()
            for excerpt in row["ordinance_text"].split(" ... "):
                assert excerpt in case["source_text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Commercial solar energy systems require setbacks.", True),
        ("Utility-scale photovoltaic solar farms require screening.", True),
        (
            (
                "Small solar energy systems are exempt. "
                "Utility-scale solar farms require setbacks."
            ),
            True,
        ),
        ("Small solar energy systems are exempt.", False),
        ("Concentrated solar power (CSP) requires a permit.", False),
        ("A solar eclipse or solar flare is not a land use.", False),
        ("This district uses land for agriculture.", False),
    ],
)
async def test_solar_one_shot_heuristic_retains_solar_scope(text, expected):
    """Keep solar farm matches while stripping misleading solar phrases"""
    plugin = COMPASSSolarOneShotExtractor(None, {})
    heuristic = await plugin.get_heuristic()
    assert heuristic.check(text) is expected


@pytest.mark.parametrize(
    "component, legacy_class",
    [
        ("TEXT_COLLECTORS", WindOrdinanceTextCollector),
        ("TEXT_EXTRACTORS", WindOrdinanceTextExtractor),
    ],
)
def test_wind_one_shot_prompts_match_legacy_wind(component, legacy_class):
    """Keep effective one-shot prompts aligned with legacy wind wording"""
    components = getattr(COMPASSWindOneShotExtractor, component)
    assert len(components) == 1
    actual = components[0].PROMPTS
    expected = legacy_class.PROMPTS
    assert len(actual) == len(expected)
    for actual_prompt, expected_prompt in zip(actual, expected, strict=True):
        assert actual_prompt.keys() == expected_prompt.keys()
        for key, value in expected_prompt.items():
            expected_value = value
            if component == "TEXT_EXTRACTORS" and key == "prompt":
                expected_value = value.format(
                    FORMATTING_PROMPT=legacy_class.FORMATTING_PROMPT,
                    OUTPUT_PROMPT=legacy_class.OUTPUT_PROMPT,
                )
            assert " ".join(actual_prompt[key].split()) == " ".join(
                expected_value.split()
            )


def test_wind_one_shot_retains_legacy_features():
    """Cover all legacy wind features with authored extraction rules"""
    schema = COMPASSWindOneShotExtractor.SCHEMA
    items = schema["properties"]["outputs"]["items"]
    features = items["properties"]["feature"]["enum"]
    expected = set(WIND_NUMERICAL_RESTRICTIONS)
    expected.update(WIND_QUALITATIVE_RESTRICTIONS)
    districts = {
        entry["feature_id"]
        for entry in StructuredWindPermittedUseDistrictsParser._USE_TYPES
    }
    expected.update(districts)
    for entry in SetbackFeatures():
        feature = entry["feature_id"]
        if feature in {"structures", "property line"}:
            expected.update(
                f"{feature} ({owner})"
                for owner in ("participating", "non-participating")
            )
        else:
            expected.add(feature)

    assert len(features) == len(set(features)) == 31
    assert set(features) == expected
    definitions = {
        feature: definition
        for family in (
            "setback_features",
            "numerical_restrictions",
            "qualitative_restrictions",
            "permitted_use_districts",
        )
        for feature, definition in schema["$definitions"][family][
            "properties"
        ].items()
    }
    assert set(definitions) == expected
    for feature, definition in definitions.items():
        assert "VALUE" in definition["description"]
        if feature in WIND_NUMERICAL_RESTRICTIONS:
            assert "UNITS:" in definition["description"]
        elif (
            feature not in WIND_QUALITATIVE_RESTRICTIONS
            and feature not in districts
        ):
            for rule in ("UNITS", "IGNORE"):
                assert rule in definition["description"]


def test_wind_one_shot_has_evidenced_examples():
    """Require complete wind examples without absence placeholders"""
    schema = COMPASSWindOneShotExtractor.SCHEMA
    items = schema["properties"]["outputs"]["items"]
    expected = set(items["properties"]["feature"]["enum"])
    districts = {
        entry["feature_id"]
        for entry in StructuredWindPermittedUseDistrictsParser._USE_TYPES
    }
    rows = schema["$examples"]
    assert rows
    assert len(rows) == len({row["feature"] for row in rows})
    for row in rows:
        assert set(row) == set(items["required"])
        assert row["feature"] in expected
        for field in ("summary", "ordinance_text", "explanation"):
            assert isinstance(row[field], str) and row[field].strip()
            assert not row[field].casefold().startswith("no explicit")
        if row["feature"] in WIND_QUALITATIVE_RESTRICTIONS:
            assert row["value"] is None
            assert row["units"] is None
        elif row["feature"] in districts:
            assert isinstance(row["value"], list) and row["value"]
            assert all(
                isinstance(district, str) and district.strip()
                for district in row["value"]
            )
            assert row["units"] is None
        else:
            assert isinstance(row["value"], (int, float))
            assert isinstance(row["units"], str) and row["units"].strip()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Commercial wind energy systems require setbacks.", True),
        ("Utility-scale wind turbines require permits.", True),
        ("Large WECS require setbacks.", True),
        (
            (
                "Small wind energy systems are exempt. "
                "Utility-scale wind farms require setbacks."
            ),
            True,
        ),
        ("Small wind energy systems are exempt.", False),
        ("Micro WECS are exempt.", False),
        ("Windows and windshields must resist wind damage.", False),
        ("This district uses land for agriculture.", False),
    ],
)
async def test_wind_one_shot_heuristic_retains_wind_scope(text, expected):
    """Keep wind farm matches while stripping misleading wind phrases"""
    plugin = COMPASSWindOneShotExtractor(None, {})
    heuristic = await plugin.get_heuristic()
    assert heuristic.check(text) is expected


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
                            "ordinance_text",
                            "explanation",
                            "section",
                            "source",
                        ],
                        "properties": {
                            "feature": {},
                            "value": {},
                            "units": {},
                            "location": {},
                            "summary": {},
                            "ordinance_text": {},
                            "explanation": {},
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
        "ordinance_text",
        "explanation",
        "section",
        "year",
        "source",
    ]
    assert "quantitative" not in [col.name for col in cols]


def test_out_cols_from_config_keeps_summary():
    """Test summary reaches the output alongside ordinance_text"""

    config = {
        "schema": {
            "properties": {
                "outputs": {
                    "items": {
                        "required": ["feature", "summary", "ordinance_text"],
                        "properties": {
                            "feature": {},
                            "summary": {},
                            "ordinance_text": {},
                        },
                    }
                }
            }
        }
    }

    col_names = [col.name for col in _out_cols_from_config(config)]

    assert "summary" in col_names
    assert "ordinance_text" in col_names


def test_out_cols_from_config_keeps_explanation():
    """Test the explanation field reaches the output columns"""

    config = {
        "schema": {
            "properties": {
                "outputs": {
                    "items": {
                        "required": ["feature", "explanation"],
                        "properties": {"feature": {}, "explanation": {}},
                    }
                }
            }
        }
    }

    assert "explanation" in [col.name for col in _out_cols_from_config(config)]


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
