"""COMPASS ordinance validation tests"""

from pathlib import Path

import pytest

from compass.extraction.wind.ordinance import WindHeuristic
from compass.extraction.solar.ordinance import SolarHeuristic


@pytest.mark.parametrize(
    "text,truth",
    [
        ("Wind SETBACKS", True),
        (" WECS SETBACKS", True),
        ("Window SETBACKS", False),
        ("SWECS SETBACKS", False),
        ("(wind LWET)", True),
        ("(wind\n LWET)", True),
        ("Wind SWECS", False),
        ("Wind WES", False),
        ("Wind WES\n", True),
        ("wind turbines and wind towers", True),
    ],
)
def test_possibly_mentions_wind(text, truth):
    """Test for `WindHeuristic` class (basic execution)"""

    assert WindHeuristic().check(text) == truth


@pytest.mark.parametrize(
    "text,truth",
    [
        ("Solar SETBACKS", True),
        (" SECS SETBACKS", True),
        ("SOLARIS SETBACKS", False),
        ("WECS SETBACKS", False),
        ("(solar farm)", True),
        ("(solar\nfarm)", True),
        ("Solar WECS", False),
        ("Solar SES", False),
        ("Solar SES\n", True),
        ("solar panels and solar farms", True),
    ],
)
def test_possibly_mentions_solar(text, truth):
    """Test for `SolarHeuristic` class (basic execution)"""

    assert SolarHeuristic().check(text) == truth


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
