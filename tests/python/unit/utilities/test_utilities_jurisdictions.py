"""COMPASS Ordinance jurisdiction utilities tests"""

from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from compass.utilities.jurisdictions import (
    load_all_jurisdiction_info,
    load_jurisdictions_from_fp,
    jurisdiction_websites,
    Jurisdiction,
    _JURISDICTION_TYPES_AS_PREFIXES,
)
from compass.exceptions import COMPASSValueError
from compass.warn import COMPASSWarning


def test_load_all_jurisdictions():
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
    for g, data in jurisdiction_info.groupby(
        ["County", "State", "Subdivision", "Jurisdiction Type"]
    ):
        if len(data) > 1:
            print(g)
            print(data)
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
    """Test that `load_jurisdictions_from_fp` returns a single county"""

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


def test_load_jurisdictions_no_repeated_counties(tmp_path):
    """Test that `load_jurisdictions_from_fp` doesn't have repeats"""

    test_jurisdiction_fp = tmp_path / "out.csv"
    input_jurisdictions = pd.DataFrame(
        {
            "County": ["Jefferson", "Jefferson", "Jefferson"],
            "State": ["Alabama", "Colorado", "Alabama"],
        }
    )
    input_jurisdictions.to_csv(test_jurisdiction_fp)

    jurisdictions = load_jurisdictions_from_fp(test_jurisdiction_fp)

    assert len(jurisdictions) == 2
    assert set(jurisdictions["County"]) == {"Jefferson"}
    assert set(jurisdictions["State"]) == {"Alabama", "Colorado"}
    assert set(jurisdictions["Subdivision"]) == {None}
    assert set(jurisdictions["Subdivision"]) != {np.nan}
    assert set(jurisdictions["Jurisdiction Type"]) == {"county"}
    assert {type(val) for val in jurisdictions["FIPS"]} == {int}


def test_load_jurisdictions_no_repeated_townships(tmp_path):
    """Test that `load_jurisdictions_from_fp` doesn't have repeats"""

    test_jurisdiction_fp = tmp_path / "out.csv"
    input_jurisdictions = pd.DataFrame(
        {
            "County": "Aroostook",
            "State": "Maine",
            "Subdivision": ["Perham", "Oakfield", "Perham"],
            "Jurisdiction Type": "town",
        }
    )
    input_jurisdictions.to_csv(test_jurisdiction_fp)

    jurisdictions = load_jurisdictions_from_fp(test_jurisdiction_fp)

    assert len(jurisdictions) == 2
    assert set(jurisdictions["County"]) == {"Aroostook"}
    assert set(jurisdictions["State"]) == {"Maine"}
    assert set(jurisdictions["Subdivision"]) == {"Perham", "Oakfield"}
    assert set(jurisdictions["Jurisdiction Type"]) == {"town"}
    assert {type(val) for val in jurisdictions["FIPS"]} == {int}


def test_load_jurisdictions_no_repeated_townships_and_counties(tmp_path):
    """Test that `load_jurisdictions_from_fp` doesn't have repeats"""

    test_jurisdiction_fp = tmp_path / "out.csv"
    input_jurisdictions = pd.DataFrame(
        {
            "County": "Aroostook",
            "State": "Maine",
            "Subdivision": ["Perham", "Oakfield", "Perham", None, None],
            "Jurisdiction Type": ["town", "town", "town", "county", "county"],
        }
    )
    input_jurisdictions.to_csv(test_jurisdiction_fp)

    jurisdictions = load_jurisdictions_from_fp(test_jurisdiction_fp)

    assert len(jurisdictions) == 3
    assert set(jurisdictions["County"]) == {"Aroostook"}
    assert set(jurisdictions["State"]) == {"Maine"}
    assert set(jurisdictions["Subdivision"]) == {"Perham", "Oakfield", None}
    assert set(jurisdictions["Jurisdiction Type"]) == {"town", "county"}
    assert {type(val) for val in jurisdictions["FIPS"]} == {int}


def test_basic_state_properties():
    """Test basic properties for ``Jurisdiction`` class for a state"""

    state = Jurisdiction("state", state="Colorado")

    assert repr(state) == "Colorado"
    assert state.full_name == "Colorado"
    assert state.full_name == str(state)

    assert not state.full_county_phrase
    assert not state.full_subdivision_phrase

    assert state == Jurisdiction("state", state="cOlORAdo")
    assert state != Jurisdiction("city", state="Colorado")

    assert state == "Colorado"
    assert state == "colorado"


def test_basic_county_properties():
    """Test basic properties for ``Jurisdiction`` class for a county"""

    county = Jurisdiction("county", county="Box Elder", state="Utah")

    assert repr(county) == "Box Elder County, Utah"
    assert county.full_name == "Box Elder County, Utah"
    assert county.full_name == str(county)

    assert county.full_county_phrase == "Box Elder County"
    assert not county.full_subdivision_phrase

    assert county != Jurisdiction("county", county="Box elder", state="uTah")
    assert county != Jurisdiction("city", county="Box Elder", state="Utah")

    assert county == "Box Elder County, Utah"
    assert county == "Box elder county, Utah"


def test_basic_parish_properties():
    """Test basic properties for ``Jurisdiction`` class for a parish"""

    parish = Jurisdiction("parish", county="Assumption", state="Louisiana")

    assert repr(parish) == "Assumption Parish, Louisiana"
    assert parish.full_name == "Assumption Parish, Louisiana"
    assert parish.full_name == str(parish)

    assert parish.full_county_phrase == "Assumption Parish"
    assert not parish.full_subdivision_phrase

    assert parish == Jurisdiction(
        "parish", county="Assumption", state="lOuisiana"
    )
    assert parish != Jurisdiction(
        "parish", county="assumption", state="lOuisiana"
    )
    assert parish != Jurisdiction(
        "county", county="Assumption", state="Louisiana"
    )

    assert parish == "Assumption Parish, Louisiana"
    assert parish == "assumption parish, lOuisiana"


@pytest.mark.parametrize("jt", ["town", "city", "borough", "township"])
def test_basic_town_properties(jt):
    """Test basic properties for ``Jurisdiction`` class for a town"""

    town = Jurisdiction(
        jt, county="Jefferson", state="Colorado", subdivision_name="Golden"
    )

    assert repr(town) == f"{jt.title()} of Golden, Jefferson County, Colorado"
    assert (
        town.full_name == f"{jt.title()} of Golden, Jefferson County, Colorado"
    )
    assert town.full_name == str(town)
    assert town.full_county_phrase == "Jefferson County"
    assert town.full_subdivision_phrase == f"{jt.title()} of Golden"

    assert town == Jurisdiction(
        jt, county="Jefferson", state="colorado", subdivision_name="Golden"
    )
    assert town != Jurisdiction(
        jt, county="jefferson", state="colorado", subdivision_name="Golden"
    )
    assert town != Jurisdiction(
        jt, county="Jefferson", state="colorado", subdivision_name="golden"
    )
    assert town != Jurisdiction(
        "county",
        county="Jefferson",
        state="Colorado",
        subdivision_name="Golden",
    )

    assert town == f"{jt.title()} of Golden, Jefferson County, Colorado"
    assert town == f"{jt.title()} of golden, jefferson county, colorado"


def test_atypical_subdivision_properties():
    """Test basic properties for ``Jurisdiction`` class for a subdivision"""

    gore = Jurisdiction(
        "gore", county="Chittenden", state="Vermont", subdivision_name="Buels"
    )

    assert repr(gore) == "Buels Gore, Chittenden County, Vermont"
    assert gore.full_name == "Buels Gore, Chittenden County, Vermont"
    assert gore.full_name == str(gore)
    assert gore.full_county_phrase == "Chittenden County"
    assert gore.full_subdivision_phrase == "Buels Gore"

    assert gore == Jurisdiction(
        "gore", county="Chittenden", state="vermont", subdivision_name="Buels"
    )
    assert gore != Jurisdiction(
        "gore", county="chittenden", state="vermont", subdivision_name="Buels"
    )
    assert gore != Jurisdiction(
        "gore", county="Chittenden", state="vermont", subdivision_name="buels"
    )
    assert gore != Jurisdiction(
        "county",
        county="Chittenden",
        state="Vermont",
        subdivision_name="Buels",
    )

    assert gore == "Buels Gore, Chittenden County, Vermont"
    assert gore == "buels gOre, chittENden county, vermonT"


def test_city_no_county():
    """Test ``Jurisdiction`` for a city with no county"""

    gore = Jurisdiction("city", "Maryland", subdivision_name="Baltimore")

    assert repr(gore) == "City of Baltimore, Maryland"
    assert gore.full_name == "City of Baltimore, Maryland"
    assert gore.full_name == str(gore)

    assert not gore.full_county_phrase
    assert gore.full_subdivision_phrase == "City of Baltimore"

    assert gore == Jurisdiction(
        "city", "maryland", subdivision_name="Baltimore"
    )
    assert gore != Jurisdiction(
        "city", "maryland", subdivision_name="baltimore"
    )
    assert gore != Jurisdiction(
        "county", "maryland", subdivision_name="baltimore"
    )

    assert gore == "City of Baltimore, Maryland"
    assert gore == "ciTy of baltiMore, maryland"


def test_full_name_the_prefixed_property():
    """Test ``Jurisdiction.full_name_the_prefixed`` property"""

    state = Jurisdiction("state", state="Colorado")
    assert state.full_name_the_prefixed == "the state of Colorado"

    county = Jurisdiction("county", state="Colorado", county="Jefferson")
    assert county.full_name_the_prefixed == "Jefferson County, Colorado"

    city = Jurisdiction(
        "city", state="Colorado", county="Jefferson", subdivision_name="Golden"
    )
    assert (
        city.full_name_the_prefixed
        == "the City of Golden, Jefferson County, Colorado"
    )

    for st in _JURISDICTION_TYPES_AS_PREFIXES:
        jur = Jurisdiction(st, state="Colorado", subdivision_name="Test")
        assert (
            jur.full_name_the_prefixed == f"the {st.title()} of Test, Colorado"
        )

    jur = Jurisdiction(st, state="Colorado", subdivision_name="Test")
    assert jur.full_name_the_prefixed == f"the {st.title()} of Test, Colorado"

    jur = Jurisdiction(
        "census county division",
        state="Colorado",
        county="Test a",
        subdivision_name="Test b",
    )

    assert (
        jur.full_name_the_prefixed
        == "Test b Census County Division, Test a County, Colorado"
    )


def test_full_subdivision_phrase_the_prefixed_property():
    """Test ``Jurisdiction.full_subdivision_phrase_the_prefixed`` property"""

    for st in _JURISDICTION_TYPES_AS_PREFIXES:
        jur = Jurisdiction(st, state="Colorado", subdivision_name="Test")
        assert (
            jur.full_subdivision_phrase_the_prefixed
            == f"the {st.title()} of Test"
        )

    jur = Jurisdiction(
        "census county division",
        state="Colorado",
        county="Test a",
        subdivision_name="Test b",
    )

    assert (
        jur.full_subdivision_phrase_the_prefixed
        == "Test b Census County Division"
    )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
