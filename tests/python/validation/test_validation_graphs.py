"""COMPASS Ordinance content validation tests"""

from pathlib import Path

import pytest

from compass.utilities.location import Jurisdiction
from compass.validation.graphs import (
    setup_graph_correct_jurisdiction_type,
    setup_graph_correct_jurisdiction_from_url,
    _jurisdiction_names_to_extract,
)


def test_setup_graph_correct_jurisdiction_type_state():
    """Test setting up jurisdiction validation graph for state"""
    loc = Jurisdiction("state", state="New York")
    graph = setup_graph_correct_jurisdiction_type(loc)

    assert set(graph.nodes) == {
        "init",
        "has_name",
        "is_state",
        "has_state_name",
        "final",
    }
    assert set(graph.edges) == {
        ("init", "has_name"),
        ("has_name", "is_state"),
        ("is_state", "has_state_name"),  # is_state --YES-> has_state_name
        ("is_state", "final"),  # is_state --NO-> has_state_name
        ("has_state_name", "final"),
    }

    assert f"{loc.state}" in graph.nodes["is_state"]["prompt"]
    assert "state" in graph.nodes["is_state"]["prompt"]
    assert "the state of" not in graph.nodes["is_state"]["prompt"].casefold()
    assert loc.full_name in graph.nodes["final"]["prompt"]


@pytest.mark.parametrize("county_type", ["parish", "county"])
def test_setup_graph_correct_jurisdiction_type_county(county_type):
    """Test setting up jurisdiction validation graph for county"""

    loc = Jurisdiction(county_type, state="New York", county="Test")
    graph = setup_graph_correct_jurisdiction_type(loc)

    assert set(graph.nodes) == {
        "init",
        "has_name",
        "is_state",
        "is_county",
        "has_county_name",
        "final",
    }
    assert set(graph.edges) == {
        ("init", "has_name"),
        ("has_name", "is_state"),
        ("is_state", "is_county"),  # is_state --NO-> is_county
        ("is_state", "final"),  # is_state --YES-> final (bad jur)
        ("is_county", "final"),  # is_county --NO-> final (bad jur)
        ("is_county", "has_county_name"),  # is_county --YES-> has_county_name
        ("has_county_name", "final"),
    }

    assert f"{loc.state}" in graph.nodes["is_state"]["prompt"]
    assert "state" in graph.nodes["is_state"]["prompt"]
    assert "the state of" not in graph.nodes["is_state"]["prompt"].casefold()

    assert loc.full_county_phrase in graph.nodes["is_county"]["prompt"]
    assert (
        f"the {loc.full_county_phrase}"
        not in graph.nodes["is_county"]["prompt"]
    )

    assert loc.full_name in graph.nodes["final"]["prompt"]


def test_setup_graph_correct_jurisdiction_type_city_no_county():
    """Test setting up jurisdiction validation graph for city no county"""

    loc = Jurisdiction("city", state="New York", subdivision_name="test")
    graph = setup_graph_correct_jurisdiction_type(loc)

    assert set(graph.nodes) == {
        "init",
        "has_name",
        "is_state",
        "is_city",
        "has_city_name",
        "final",
    }
    assert set(graph.edges) == {
        ("init", "has_name"),
        ("has_name", "is_state"),
        ("is_state", "final"),  # is_state --YES-> final (bad jur)
        ("is_state", "is_city"),  # is_state --NO-> is_county
        ("is_city", "final"),  # is_city --NO-> final (bad jur)
        ("is_city", "has_city_name"),  # is_city --YES-> has_city_name
        ("has_city_name", "final"),
    }

    assert f"{loc.state}" in graph.nodes["is_state"]["prompt"]
    assert "state" in graph.nodes["is_state"]["prompt"]
    assert "the state of" not in graph.nodes["is_state"]["prompt"].casefold()

    assert (
        f"the {loc.full_subdivision_phrase}"
        in graph.nodes["is_city"]["prompt"]
    )

    assert loc.full_name in graph.nodes["final"]["prompt"]


def test_setup_graph_correct_jurisdiction_type_city():
    """Test setting up jurisdiction validation graph for city"""

    loc = Jurisdiction(
        "city", state="Colorado", county="Jefferson", subdivision_name="Golden"
    )
    graph = setup_graph_correct_jurisdiction_type(loc)

    assert set(graph.nodes) == {
        "init",
        "has_name",
        "is_state",
        "is_county",
        "is_city",
        "has_city_name",
        "final",
    }
    assert set(graph.edges) == {
        ("init", "has_name"),
        ("has_name", "is_state"),
        ("is_state", "final"),  # is_state --YES-> final (bad jur)
        ("is_state", "is_county"),  # is_state --NO-> is_county
        ("is_county", "final"),  # is_county --YES-> final (bad jur)
        ("is_county", "is_city"),  # is_county --NO-> is_city
        ("is_city", "final"),  # is_city --NO-> final (bad jur)
        ("is_city", "has_city_name"),  # is_city --YES-> has_city_name
        ("has_city_name", "final"),
    }

    assert f"{loc.state}" in graph.nodes["is_state"]["prompt"]
    assert "state" in graph.nodes["is_state"]["prompt"]
    assert "the state of" not in graph.nodes["is_state"]["prompt"].casefold()

    assert loc.full_county_phrase in graph.nodes["is_county"]["prompt"]
    assert (
        f"the {loc.full_subdivision_phrase}"
        in graph.nodes["is_city"]["prompt"]
    )

    assert loc.full_name in graph.nodes["final"]["prompt"]


def test_setup_graph_correct_jurisdiction_from_url_state():
    """Test setting up URL jurisdiction validation graph for state"""

    loc = Jurisdiction("state", state="New York")
    graph = setup_graph_correct_jurisdiction_from_url(loc)

    assert set(graph.nodes) == {"init", "final"}
    assert list(graph.edges) == [("init", "final")]

    assert f"{loc.state} state" in graph.nodes["init"]["prompt"]
    assert "correct_state" in graph.nodes["final"]["prompt"]
    assert f"{loc.state} state" in graph.nodes["final"]["prompt"]


@pytest.mark.parametrize("county_type", ["parish", "county"])
def test_setup_graph_correct_jurisdiction_from_url_county(county_type):
    """Test setting up URL jurisdiction validation graph for county"""

    loc = Jurisdiction(county_type, state="New York", county="Test")
    graph = setup_graph_correct_jurisdiction_from_url(loc)

    assert set(graph.nodes) == {"init", "mentions_county", "final"}
    assert list(graph.edges) == [
        ("init", "mentions_county"),
        ("mentions_county", "final"),
    ]

    assert f"{loc.state} state" in graph.nodes["init"]["prompt"]

    assert loc.full_county_phrase in graph.nodes["mentions_county"]["prompt"]
    assert (
        f"the {loc.full_county_phrase}"
        not in graph.nodes["mentions_county"]["prompt"]
    )

    assert "correct_state" in graph.nodes["final"]["prompt"]
    assert f"{loc.state} state" in graph.nodes["final"]["prompt"]
    assert "the state of" not in graph.nodes["final"]["prompt"].casefold()

    assert "correct_county" in graph.nodes["final"]["prompt"]
    assert loc.full_county_phrase in graph.nodes["final"]["prompt"]
    assert (
        f"the {loc.full_county_phrase}" not in graph.nodes["final"]["prompt"]
    )


def test_setup_graph_correct_jurisdiction_from_url_city():
    """Test setting up URL jurisdiction validation graph for city"""

    loc = Jurisdiction(
        "city", state="Colorado", county="Jefferson", subdivision_name="Golden"
    )
    graph = setup_graph_correct_jurisdiction_from_url(loc)

    assert set(graph.nodes) == {
        "init",
        "mentions_county",
        "mentions_city",
        "final",
    }
    assert list(graph.edges) == [
        ("init", "mentions_county"),
        ("mentions_county", "mentions_city"),
        ("mentions_city", "final"),
    ]

    assert f"{loc.state} state" in graph.nodes["init"]["prompt"]

    assert loc.full_county_phrase in graph.nodes["mentions_county"]["prompt"]
    assert (
        f"the {loc.full_county_phrase}"
        not in graph.nodes["mentions_county"]["prompt"]
    )

    assert (
        f"the {loc.full_subdivision_phrase}"
        in graph.nodes["mentions_city"]["prompt"]
    )

    assert "correct_state" in graph.nodes["final"]["prompt"]
    assert f"{loc.state} state" in graph.nodes["final"]["prompt"]

    assert "correct_county" in graph.nodes["final"]["prompt"]
    assert loc.full_county_phrase in graph.nodes["final"]["prompt"]

    assert "correct_city" in graph.nodes["final"]["prompt"]
    assert loc.full_subdivision_phrase in graph.nodes["final"]["prompt"]


def test_setup_graph_correct_jurisdiction_from_url_gore():
    """Test setting up URL jurisdiction validation graph for gore"""

    loc = Jurisdiction("gore", state="Vermont", subdivision_name="Buels")
    graph = setup_graph_correct_jurisdiction_from_url(loc)

    assert set(graph.nodes) == {"init", "mentions_city", "final"}
    assert list(graph.edges) == [
        ("init", "mentions_city"),
        ("mentions_city", "final"),
    ]

    assert f"{loc.state} state" in graph.nodes["init"]["prompt"]
    assert (
        loc.full_subdivision_phrase in graph.nodes["mentions_city"]["prompt"]
    )
    assert (
        f"the {loc.full_subdivision_phrase}"
        not in graph.nodes["mentions_city"]["prompt"]
    )

    assert "correct_state" in graph.nodes["final"]["prompt"]
    assert f"{loc.state} state" in graph.nodes["final"]["prompt"]
    assert "the state of" not in graph.nodes["final"]["prompt"].casefold()

    assert "correct_county" not in graph.nodes["final"]["prompt"]

    assert "correct_gore" in graph.nodes["final"]["prompt"]
    assert loc.full_subdivision_phrase in graph.nodes["final"]["prompt"]


@pytest.mark.parametrize(
    "loc,expected_text",
    [
        (
            Jurisdiction(
                "town",
                state="Minnesota",
                subdivision_name="Jefferson",
            ),
            "the state name and the town name",
        ),
        (
            Jurisdiction(
                "township",
                state="Minnesota",
                subdivision_name="Jefferson",
            ),
            "the state name and the township name",
        ),
        (
            Jurisdiction(
                "unincorporated area",
                state="Minnesota",
                county="Norman",
                subdivision_name="Jefferson",
            ),
            "the state name and the unincorporated area name",
        ),
        (
            Jurisdiction(
                "county",
                state="Minnesota",
                county="Norman",
            ),
            "the state name and the county name",
        ),
        (
            Jurisdiction(
                "borough",
                state="Minnesota",
                county="Norman",
            ),
            "the state name and the borough name",
        ),
        (
            Jurisdiction(
                "state",
                state="Minnesota",
            ),
            "the state name",
        ),
    ],
)
def test_jurisdiction_names_to_extract(loc, expected_text):
    """Test the `_jurisdiction_names_to_extract` function"""
    assert _jurisdiction_names_to_extract(loc) == expected_text


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
