"""COMPASS Ordinance jurisdiction utilities tests"""

from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from compass.utilities.jurisdictions import (
    load_all_jurisdiction_info,
    load_jurisdictions_from_fp,
    jurisdiction_websites,
)
from compass.exceptions import COMPASSValueError
from compass.warn import COMPASSWarning


def test_load_jurisdictions():
    """Test the `load_all_jurisdiction_info` function"""

    jurisdiction_info = load_all_jurisdiction_info()
    assert not jurisdiction_info.empty

    expected_cols = [
        "County",
        "State",
        "Subdivision",
        "Jurisdiction Type",
        "FIPS",
        "Website",
    ]
    assert all(col in jurisdiction_info for col in expected_cols)
    assert len(jurisdiction_info) == len(
        jurisdiction_info.groupby(
            ["County", "State", "Subdivision", "Jurisdiction Type"]
        )
    )
    assert len(jurisdiction_info) == len(jurisdiction_info.groupby(["FIPS"]))

    # Spot checks:
    assert "Decatur" in set(jurisdiction_info["County"])
    assert "Box Elder" in set(jurisdiction_info["County"])
    assert "Colorado" in set(jurisdiction_info["State"])
    assert "Rhode Island" in set(jurisdiction_info["State"])


def test_jurisdiction_websites():
    """Test the `jurisdiction_websites` function"""

    websites = jurisdiction_websites()
    assert len(websites) == len(load_all_jurisdiction_info())
    assert isinstance(websites, dict)

    # Spot checks:
    assert 18031 in websites  # Decatur Indiana
    assert 8041 in websites  # El Paso, Colorado
    assert 49003 in websites  # Box Elder, Utah


def test_load_jurisdictions_from_fp(tmp_path):
    """Test `load_jurisdictions_from_fp` function"""

    test_jurisdiction_fp = tmp_path / "out.csv"
    input_jurisdictions = pd.DataFrame(
        {"County": ["decatur", "DNE County"], "State": ["INDIANA", "colorado"]}
    )
    input_jurisdictions.to_csv(test_jurisdiction_fp)

    with pytest.warns(COMPASSWarning) as record:
        jurisdictions = load_jurisdictions_from_fp(test_jurisdiction_fp)

    assert len(record) == 1
    warning_msg = str(record[0].message)

    assert "nan" not in warning_msg
    assert "DNE County" in warning_msg
    assert "colorado" in warning_msg

    assert len(jurisdictions) == 1
    assert set(jurisdictions["County"]) == {"Decatur"}
    assert set(jurisdictions["State"]) == {"Indiana"}
    assert set(jurisdictions["Subdivision"]) == {None}
    assert set(jurisdictions["Subdivision"]) != {np.nan}
    assert set(jurisdictions["Jurisdiction Type"]) == {"county"}
    assert {type(val) for val in jurisdictions["FIPS"]} == {int}


def test_load_jurisdictions_from_fp_bad_input(tmp_path):
    """Test `load_jurisdictions_from_fp` function"""

    test_jurisdiction_fp = tmp_path / "out.csv"
    pd.DataFrame().to_csv(test_jurisdiction_fp)

    with pytest.raises(COMPASSValueError) as err:
        load_jurisdictions_from_fp(test_jurisdiction_fp)

    expected_msg = (
        "The jurisdiction input must have at least a 'State' column!"
    )
    assert expected_msg in str(err)


def test_load_jurisdictions_from_fp_single_county(tmp_path):
    """Test that`load_jurisdictions_from_fp` returns a single county"""

    test_jurisdiction_fp = tmp_path / "out.csv"
    input_jurisdictions = pd.DataFrame(
        {"County": ["Wharton"], "State": ["Texas"]}
    )
    input_jurisdictions.to_csv(test_jurisdiction_fp)

    jurisdictions = load_jurisdictions_from_fp(test_jurisdiction_fp)

    assert len(jurisdictions) == 1
    assert set(jurisdictions["County"]) == {"Wharton"}
    assert set(jurisdictions["State"]) == {"Texas"}
    assert set(jurisdictions["Subdivision"]) == {None}
    assert set(jurisdictions["Subdivision"]) != {np.nan}
    assert set(jurisdictions["Jurisdiction Type"]) == {"county"}
    assert {type(val) for val in jurisdictions["FIPS"]} == {int}


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
