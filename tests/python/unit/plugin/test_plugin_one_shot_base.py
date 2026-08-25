"""COMPASS one-shot plugin configuration tests"""

import json
from pathlib import Path

import pytest

from compass.plugin.one_shot.base import (
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
