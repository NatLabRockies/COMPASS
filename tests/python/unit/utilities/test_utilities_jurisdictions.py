"""COMPASS Ordinance jurisdiction utilities tests"""

from pathlib import Path

import pytest
import pandas as pd

from compass.utilities.jurisdictions import (
    jurisdiction_websites,
    jurisdictions_from_df,
    Jurisdiction,
    _JURISDICTION_TYPES_AS_PREFIXES,
)


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


def test_jurisdictions_from_df_basic():
    """Test ``jurisdictions_from_df`` generator with various row types"""

    jurisdictions_df = pd.DataFrame(
        {
            "Jurisdiction Type": ["state", "county", "city"],
            "State": ["Colorado", "Utah", "Texas"],
            "County": [None, "Box Elder", "Travis"],
            "Subdivision": [None, None, "Austin"],
            "FIPS": [8, 49003, 48453],
            "Website": [
                "https://colorado.gov",
                "https://boxeldercounty.org",
                "https://austintexas.gov",
            ],
        }
    )

    jurisdictions = list(jurisdictions_from_df(jurisdictions_df))

    assert len(jurisdictions) == 3

    state_jur = jurisdictions[0]
    assert state_jur.type == "State"
    assert state_jur.state == "Colorado"
    assert state_jur.county is None
    assert state_jur.subdivision_name is None
    assert state_jur.code == "8"
    assert state_jur.website_url == "https://colorado.gov"
    assert state_jur.full_name == "Colorado"

    county_jur = jurisdictions[1]
    assert county_jur.type == "County"
    assert county_jur.state == "Utah"
    assert county_jur.county == "Box Elder"
    assert county_jur.subdivision_name is None
    assert county_jur.code == "49003"
    assert county_jur.website_url == "https://boxeldercounty.org"
    assert county_jur.full_name == "Box Elder County, Utah"

    city_jur = jurisdictions[2]
    assert city_jur.type == "City"
    assert city_jur.state == "Texas"
    assert city_jur.county == "Travis"
    assert city_jur.subdivision_name == "Austin"
    assert city_jur.code == "48453"
    assert city_jur.website_url == "https://austintexas.gov"
    assert city_jur.full_name == "City of Austin, Travis County, Texas"


def test_jurisdictions_from_df_with_none_values():
    """Test ``jurisdictions_from_df`` handles None/missing values properly"""

    jurisdictions_df = pd.DataFrame(
        {
            "Jurisdiction Type": ["county"],
            "State": ["Indiana"],
            "County": ["Decatur"],
            "Subdivision": [None],
            "FIPS": [18031],
            "Website": [None],
        }
    )

    jurisdictions = list(jurisdictions_from_df(jurisdictions_df))

    assert len(jurisdictions) == 1
    jur = jurisdictions[0]
    assert jur.type == "County"
    assert jur.state == "Indiana"
    assert jur.county == "Decatur"
    assert jur.subdivision_name is None
    assert jur.code == "18031"
    assert jur.website_url is None


def test_jurisdictions_from_df_texas_water_districts():
    """Test ``jurisdictions_from_df`` with Texas water district pattern"""

    jurisdictions_df = pd.DataFrame(
        {
            "Jurisdiction Type": [
                "Authority & Groundwater District",
                "Aquifer Conservation District",
            ],
            "State": ["Texas", "Texas"],
            "County": [None, None],
            "Subdivision": [
                "Bandera County River",
                "Barton Springs/Edwards",
            ],
            "FIPS": [1, 2],
            "Website": [
                "https://bcragd.org",
                "https://www.bseacd.org",
            ],
        }
    )

    jurisdictions = list(jurisdictions_from_df(jurisdictions_df))

    assert len(jurisdictions) == 2

    district1 = jurisdictions[0]
    assert district1.type == "Authority & Groundwater District"
    assert district1.state == "Texas"
    assert district1.county is None
    assert district1.subdivision_name == "Bandera County River"
    assert district1.code == "1"
    assert (
        district1.full_name
        == "Bandera County River Authority & Groundwater District, Texas"
    )

    district2 = jurisdictions[1]
    assert district2.type == "Aquifer Conservation District"
    assert district2.state == "Texas"
    assert district2.county is None
    assert district2.subdivision_name == "Barton Springs/Edwards"
    assert district2.code == "2"
    assert (
        district2.full_name
        == "Barton Springs/Edwards Aquifer Conservation District, Texas"
    )


def test_jurisdiction_equality_with_non_string_non_jurisdiction():
    """Test ``Jurisdiction.__eq__`` with incompatible types returns False"""

    jur = Jurisdiction("county", state="Colorado", county="Jefferson")

    assert jur is not None
    assert jur != 42
    assert jur != ["Jefferson County", "Colorado"]
    assert jur != {"county": "Jefferson", "state": "Colorado"}


def test_jurisdiction_hash_consistency():
    """Test that equal jurisdictions have the same hash"""

    jur1 = Jurisdiction("county", state="Colorado", county="Jefferson")
    jur2 = Jurisdiction("county", state="colorado", county="Jefferson")
    jur3 = Jurisdiction("County", state="COLORADO", county="Jefferson")

    assert hash(jur1) == hash(jur2) == hash(jur3)

    jur_set = {jur1, jur2, jur3}
    assert len(jur_set) == 1


def test_jurisdiction_hash_different_types_same_name():
    """Test that jurisdictions hash correctly"""

    county = Jurisdiction("county", state="Virginia", county="Alexandria")
    city = Jurisdiction(
        "city", state="Virginia", subdivision_name="Alexandria"
    )

    assert county.full_name != city.full_name
    assert hash(county) != hash(city)


def test_full_county_phrase_with_subdivision():
    """Test ``full_county_phrase`` when subdivision exists and county exists"""

    jur = Jurisdiction(
        "town",
        state="Maine",
        county="Aroostook",
        subdivision_name="Perham",
    )

    assert jur.full_county_phrase == "Aroostook County"
    assert jur.full_subdivision_phrase == "Town of Perham"
    assert jur.full_name == "Town of Perham, Aroostook County, Maine"


def test_full_subdivision_phrase_non_prefix_type():
    """Test ``full_subdivision_phrase`` with non-prefix jurisdiction type"""

    gore = Jurisdiction(
        "gore", state="Vermont", county="Chittenden", subdivision_name="Buels"
    )

    assert gore.full_subdivision_phrase == "Buels Gore"
    assert "gore" not in _JURISDICTION_TYPES_AS_PREFIXES


def test_full_name_the_prefixed_non_prefix_type():
    """Test ``full_name_the_prefixed`` for non-state, non-prefix types"""

    parish = Jurisdiction("parish", state="Louisiana", county="Assumption")
    assert parish.full_name_the_prefixed == "Assumption Parish, Louisiana"

    gore = Jurisdiction(
        "gore", state="Vermont", county="Chittenden", subdivision_name="Buels"
    )
    assert (
        gore.full_name_the_prefixed == "Buels Gore, Chittenden County, Vermont"
    )


def test_jurisdiction_websites_custom_dataframe():
    """Test ``jurisdiction_websites`` with explicitly passed DataFrame"""

    custom_df = pd.DataFrame(
        {
            "County": ["Test County", "Another County"],
            "State": ["Colorado", "Utah"],
            "Subdivision": [None, None],
            "Jurisdiction Type": ["county", "county"],
            "FIPS": [99001, 99002],
            "Website": ["https://test.gov", "https://another.gov"],
        }
    )

    websites = jurisdiction_websites(jurisdiction_info=custom_df)

    assert len(websites) == 2
    assert websites[99001] == "https://test.gov"
    assert websites[99002] == "https://another.gov"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
