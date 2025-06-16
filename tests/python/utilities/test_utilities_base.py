"""Test COMPASS Ordinance logging logic."""

from pathlib import Path

import pytest

from compass.utilities.base import title_preserving_caps


def test_title_preserving_caps():
    """Test the `title_preserving_caps` function"""

    assert title_preserving_caps("hello world") == "Hello World"
    assert title_preserving_caps("hello World") == "Hello World"
    assert title_preserving_caps("Hello world") == "Hello World"
    assert title_preserving_caps("Hello World") == "Hello World"
    assert title_preserving_caps("HELLO WORLD") == "HELLO WORLD"

    assert title_preserving_caps("St. mcLean") == "St. McLean"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
