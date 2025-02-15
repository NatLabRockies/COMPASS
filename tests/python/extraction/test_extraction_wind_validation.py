"""COMPASS wind ordinance validation tests"""

from pathlib import Path

import pytest

from compass.extraction.wind import possibly_mentions_wind


@pytest.mark.parametrize(
    "text,truth",
    [
        ("Wind SETBACKS", True),
        (" WECS SETBACKS", True),
        ("Window SETBACKS", False),
        ("SWECS SETBACKS", False),
        ("(wind LWET)", True),
        ("Wind SWECS", False),
        ("Wind WES", False),
        ("Wind WES\n", True),
        ("wind turbines and wind towers", True),
    ],
)
def test_possibly_mentions_wind(text, truth):
    """Test for `possibly_mentions_wind` function (basic execution)"""

    assert possibly_mentions_wind(text) == truth


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
