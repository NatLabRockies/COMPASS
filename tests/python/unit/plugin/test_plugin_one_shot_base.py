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


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
