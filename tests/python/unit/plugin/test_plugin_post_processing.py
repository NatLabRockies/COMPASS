"""COMPASS plugin post-processing tests"""

from pathlib import Path

import pandas as pd
import pytest

from compass.extraction.data_centers import COMPASSDataCentersExtractor
from compass.utilities.jurisdictions import Jurisdiction
from compass.plugin.post_processing import normalize_data_center_types


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


def test_data_center_export_preserves_codes_and_normalizes_types(tmp_path):
    """Apply type rules and keep FIPS codes in the combined export"""
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
