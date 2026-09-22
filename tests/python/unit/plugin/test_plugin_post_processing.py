"""COMPASS plugin post-processing tests"""

from pathlib import Path

import pandas as pd
import pytest

from compass.plugin.post_processing import (
    MAX_ORDINANCE_TEXT_CHARS,
    POST_PROCESSING_REGISTRY,
    trim_ordinance_text,
)


def test_trim_ordinance_text_leaves_short_text_alone():
    """Test excerpts within the limit are untouched"""

    text = "Turbines shall not exceed 100 feet."
    db = pd.DataFrame([{"ordinance_text": text}])

    out = trim_ordinance_text(db)

    assert out.iloc[0]["ordinance_text"] == text


def test_trim_ordinance_text_trims_long_text():
    """Test over-long excerpts are cut back and marked"""

    long_text = "word " * (MAX_ORDINANCE_TEXT_CHARS // 2)
    db = pd.DataFrame([{"ordinance_text": long_text}])

    out = trim_ordinance_text(db)
    trimmed = out.iloc[0]["ordinance_text"]

    assert len(trimmed) <= MAX_ORDINANCE_TEXT_CHARS + len(" ...")
    assert trimmed.endswith(" ...")
    # cut on a word boundary, so no partial word is left behind
    assert not trimmed.removesuffix(" ...").endswith("wor")


def test_trim_ordinance_text_preserves_non_strings():
    """Test null entries survive trimming"""

    db = pd.DataFrame(
        [{"ordinance_text": None}, {"ordinance_text": "short text"}]
    )

    out = trim_ordinance_text(db)

    assert out.iloc[0]["ordinance_text"] is None
    assert out.iloc[1]["ordinance_text"] == "short text"


def test_trim_ordinance_text_without_column():
    """Test databases lacking the column pass through unchanged"""

    db = pd.DataFrame([{"feature": "Height"}])

    out = trim_ordinance_text(db)

    assert list(out.columns) == ["feature"]


def test_trim_ordinance_text_with_empty_db():
    """Test empty databases pass through unchanged"""

    db = pd.DataFrame(columns=["ordinance_text"])

    assert trim_ordinance_text(db).empty


def test_trim_ordinance_text_is_registered():
    """Test the step is discoverable as a post-processing step"""

    assert POST_PROCESSING_REGISTRY["trim_ordinance_text"] is (
        trim_ordinance_text
    )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
