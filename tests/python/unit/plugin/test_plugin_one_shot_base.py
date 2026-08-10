"""COMPASS one-shot plugin configuration tests"""

from pathlib import Path

import pytest

from compass.plugin.one_shot.base import _out_cols_from_config


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
        "ordinance_text",
        "explanation",
        "section",
        "year",
        "source",
        "quantitative",
    ]


def test_out_cols_from_config_drops_deprecated_summary():
    """Test the deprecated summary field is kept out of the output"""

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

    assert "summary" not in col_names
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
