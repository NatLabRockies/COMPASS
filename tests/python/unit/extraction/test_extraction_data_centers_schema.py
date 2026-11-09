"""Checks for the data center extraction schema contract"""

import json
from pathlib import Path

import compass


SCHEMA_FP = (
    Path(compass.__file__).parent
    / "extraction"
    / "data_centers"
    / "data_centers_schema.json"
)


def test_schema_asks_for_extraction_notes():
    """Every extraction explains itself, even when it finds nothing"""
    schema = json.loads(SCHEMA_FP.read_text(encoding="utf-8"))
    assert schema["required"] == ["outputs", "extraction_notes"]
    notes = schema["properties"]["extraction_notes"]
    assert notes["type"] == "string"
    assert "outputs is empty" in notes["description"]
    assert schema["additionalProperties"] is False


def test_schema_extracts_definitions_of_data_center_terms():
    """Definitions are a feature, described and marked qualitative"""
    schema = json.loads(SCHEMA_FP.read_text(encoding="utf-8"))
    features = schema["properties"]["outputs"]["items"]["properties"][
        "feature"
    ]["enum"]
    assert "definition" in features
    described = {
        name for group in schema["$definitions"].values() for name in group
    }
    assert described == set(features)
    assert "definition" in schema["$qualitative_features"]
    assert any(
        f"{len(features)} allowed feature IDs" in item
        for item in schema["$qa_checklist"]
    )


def test_numeric_features_ask_for_most_restrictive_value():
    """Numeric rows hold one value; the rest belong in the summary"""
    schema = json.loads(SCHEMA_FP.read_text(encoding="utf-8"))
    assert "compound" not in json.dumps(schema).casefold()
    assert "numeric_prioritization" in schema["$instructions"]

    qualitative = set(schema["$qualitative_features"])
    definitions = {
        name: spec["description"]
        for group in schema["$definitions"].values()
        for name, spec in group.items()
    }
    # a definition row's value is the definition text, not a requirement
    definitions.pop("definition")
    for name, description in definitions.items():
        value_rule = description.split("VALUE:")[1].split("UNITS:")[0]
        if name in qualitative:
            assert value_rule.strip().startswith("null"), name
        elif not name.endswith("districts"):
            assert "largest" in value_rule or "lowest" in value_rule, name


def test_units_use_canonical_vocabulary_without_detail():
    """Units name a canonical unit; any other detail goes in summary"""
    schema = json.loads(SCHEMA_FP.read_text(encoding="utf-8"))
    vocabulary = schema["$instructions"]["units"]
    assert any("canonical vocabulary" in item for item in vocabulary)

    for group in schema["$definitions"].values():
        for name, spec in group.items():
            units_rule = spec["description"].split("UNITS:")[1]
            units_rule = units_rule.split("SUMMARY:")[0].split("IGNORE:")[0]
            assert "exactly as written" not in units_rule, name
            if name.endswith("districts"):
                assert units_rule.strip() == "null.", name
