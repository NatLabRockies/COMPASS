"""Tests for version access

Ensures that `compass.__version__` is exposed and semantically shaped.
We allow dev/local forms produced by setuptools_scm
(e.g., 0.11.3.dev8+gHASH).
"""

import re
from pathlib import Path

import pytest

import compass


SEMVER_DEV_PATTERN = re.compile(
    r"^\d+\.\d+\.\d+(?:\.dev\d+\+g[0-9A-Fa-f]+(?:\.d\d{8})?)?$"
)


def test_version_string_present():
    """`__version__` attribute exists, is a non-empty string"""
    assert hasattr(compass, "__version__")
    assert isinstance(compass.__version__, str)
    assert compass.__version__  # non-empty


def test_version_semantic_shape():
    """Version matches dev semver pattern"""
    v = compass.__version__
    assert SEMVER_DEV_PATTERN.match(v) is not None, (
        f"Version {v} does not match expected pattern"
    )
    assert v != "9999", "Version set to placeholder"
    assert not v.startswith("10000"), "Version set to placeholder"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
