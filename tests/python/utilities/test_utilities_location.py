"""COMPASS Ordinance Location utility tests"""

from pathlib import Path

import pytest

from compass.utilities.location import Jurisdiction


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

    assert county == Jurisdiction("county", county="Box elder", state="uTah")
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
        jt, county="jefferson", state="colorado", subdivision_name="golden"
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
        "gore", county="chittenden", state="vermont", subdivision_name="buels"
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
        "city", "maryland", subdivision_name="baltimore"
    )
    assert gore != Jurisdiction(
        "county", "maryland", subdivision_name="baltimore"
    )

    assert gore == "City of Baltimore, Maryland"
    assert gore == "ciTy of baltiMore, maryland"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
