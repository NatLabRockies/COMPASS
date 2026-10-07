"""COMPASS one-shot plugin configuration tests"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd
import pyjson5

from compass.plugin.one_shot.base import _out_cols_from_config
from compass.plugin.one_shot.components import SchemaOrdinanceParser
from compass.utilities.finalize import doc_infos_to_db, save_db
from compass.utilities.jurisdictions import Jurisdiction


@pytest.mark.parametrize(
    "schema_path",
    [
        "compass/extraction/geothermal_electricity/geothermal_schema.json",
        "compass/extraction/ghp/geothermal_heat_pump_schema.json5",
        "compass/extraction/rmp/rmp_schema.json",
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


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
