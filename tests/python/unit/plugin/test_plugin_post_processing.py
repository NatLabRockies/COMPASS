"""COMPASS plugin post-processing tests"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from compass._cli.finalize import _compile_db
from compass.extraction.data_centers import COMPASSDataCentersExtractor
from compass.plugin.registry import PLUGIN_REGISTRY
from compass.utilities.jurisdictions import Jurisdiction
from compass.plugin.post_processing import (
    MAX_ORDINANCE_TEXT_CHARS,
    POST_PROCESSING_REGISTRY,
    normalize_data_center_types,
    trim_ordinance_text,
)


def test_trim_ordinance_text_leaves_short_text_alone():
    """Test excerpts within the limit are untouched"""

    text = "Turbines shall not exceed 100 feet."
    db = pd.DataFrame([{"ordinance_text": text}])

    out = trim_ordinance_text(db)

    assert out.iloc[0]["ordinance_text"] == text


def test_trim_ordinance_text_trims_long_text():
    """Test over-long excerpts are cut back and marked"""

    long_text = "word " * (MAX_ORDINANCE_TEXT_CHARS // 2)
    db = pd.DataFrame([{"ordinance_text": long_text}])

    out = trim_ordinance_text(db)
    trimmed = out.iloc[0]["ordinance_text"]

    assert len(trimmed) <= MAX_ORDINANCE_TEXT_CHARS + len(" ...")
    assert trimmed.endswith(" ...")
    # cut on a word boundary, so no partial word is left behind
    assert not trimmed.removesuffix(" ...").endswith("wor")


def test_trim_ordinance_text_preserves_non_strings():
    """Test null entries survive trimming"""

    db = pd.DataFrame(
        [{"ordinance_text": None}, {"ordinance_text": "short text"}]
    )

    out = trim_ordinance_text(db)

    assert out.iloc[0]["ordinance_text"] is None
    assert out.iloc[1]["ordinance_text"] == "short text"


def test_trim_ordinance_text_without_column():
    """Test databases lacking the column pass through unchanged"""

    db = pd.DataFrame([{"feature": "Height"}])

    out = trim_ordinance_text(db)

    assert list(out.columns) == ["feature"]


def test_trim_ordinance_text_with_empty_db():
    """Test empty databases pass through unchanged"""

    db = pd.DataFrame(columns=["ordinance_text"])

    assert trim_ordinance_text(db).empty


def test_trim_ordinance_text_is_registered():
    """Test the step is discoverable as a post-processing step"""

    assert POST_PROCESSING_REGISTRY["trim_ordinance_text"] is (
        trim_ordinance_text
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (None, None),
        ("", None),
        ("Data Center", None),
        ("DATA Center", None),
        ("  DATA   CENTER ", None),
        ("Data Centers", None),
        ("All data center types", None),
        ("Accessory Data Center ", "Accessory Data Center"),
        (
            (
                "Accessory Data Center "
                "(as Accessory to an Office or Industrial Use)"
            ),
            "Accessory Data Center",
        ),
        ("accessory data center (office use)", "Accessory Data Center"),
        ("Hyperscale Data Center", "Hyperscale Data Center"),
        ("Micro Data Center", "Micro Data Center"),
        ("Data Center, Medium", "Data Center, Medium"),
        ("Data Center (Micro)", "Data Center (Micro)"),
    ],
)
def test_normalize_data_center_types(label, expected):
    """Normalize generic and accessory labels without changing evidence"""
    db = pd.DataFrame(
        [{"data_center_type": label, "ordinance_text": "Original wording"}]
    )

    out = normalize_data_center_types(db)

    assert out.loc[0, "data_center_type"] == expected
    assert out.loc[0, "ordinance_text"] == "Original wording"


@pytest.mark.parametrize("finalize", [False, True])
def test_data_center_export_preserves_codes_and_normalizes_types(
    tmp_path, monkeypatch, finalize
):
    """Apply the same type rules in normal exports and CLI finalization"""
    monkeypatch.setitem(
        PLUGIN_REGISTRY, "data_centers", COMPASSDataCentersExtractor
    )
    jurisdiction = Jurisdiction(
        "city", "Arizona", subdivision_name="Mesa", code="0446000"
    )
    source_dir = tmp_path / "jurisdiction_dbs"
    source_dir.mkdir()
    source_fp = source_dir / f"{jurisdiction.full_name} Ordinances.csv"
    labels = [
        "DATA CENTER",
        "Accessory Data Center (as Accessory to an Office or Industrial Use)",
        "Hyperscale Data Center",
    ]
    pd.DataFrame(
        [
            {
                "feature": "maximum height",
                "data_center_type": label,
                "value": 40,
                "units": "feet",
                "summary": "Maximum height is 40 feet.",
                "ordinance_text": "Original wording",
                "explanation": "The text states a height limit.",
                "quantitative": True,
                "year": 2026,
                "source": "https://example.com/ordinance.pdf",
            }
            for label in labels
        ]
    ).to_csv(source_fp, index=False)

    if finalize:
        _compile_db(
            [
                {
                    "found": True,
                    "jurisdiction_type": "city",
                    "state": "Arizona",
                    "subdivision": "Mesa",
                    "FIPS": "0446000",
                    "documents": [{"source": "ordinance.pdf"}],
                }
            ],
            SimpleNamespace(out=tmp_path, jurisdiction_dbs=source_dir),
            "data_centers",
        )
    else:
        COMPASSDataCentersExtractor.save_structured_data(
            [{"ord_db_fp": source_fp, "jurisdiction": jurisdiction}],
            tmp_path,
        )

    out = pd.read_csv(
        tmp_path / "ordinances.csv", dtype=str, keep_default_na=False
    )
    assert out["FIPS"].tolist() == ["0446000"] * len(labels)
    assert out["data_center_type"].tolist() == [
        "",
        "Accessory Data Center",
        "Hyperscale Data Center",
    ]
    assert out["ordinance_text"].tolist() == ["Original wording"] * len(labels)


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
