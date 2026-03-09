"""Test COMPASS Ordinance parsing utilities"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.utilities.parsing import (
    clean_backticks_from_llm_response,
    convert_paths_to_strings,
    extract_year_from_doc_attrs,
    llm_response_as_json,
    merge_overlapping_texts,
    num_ordinances_dataframe,
    ordinances_bool_index,
)


@pytest.mark.parametrize(
    "in_str,expected",
    [
        ("plain text", "plain text"),
        ("```code```", "code"),
        ("```\ncode\n```", "code\n"),
        ("  ```json\ncode```  ", "json\ncode"),
        ("\n```\ncode\n```\n", "code\n"),
        ("\r\n```\r\ncode\r\n```\r\n", "\r\ncode\r\n"),
        ("```", ""),
    ],
)
def test_clean_backticks_from_llm_response(in_str, expected):
    """Test the `clean_backticks_from_llm_response` function"""

    assert clean_backticks_from_llm_response(in_str) == expected


@pytest.mark.parametrize(
    "in_str,expected",
    [
        (' {"a": 1} ', {"a": 1}),
        ('```json\n{"a": True, "b": False}```', {"a": True, "b": False}),
        ('{"a": True', {}),
        ('json\n{"key": "value"}', {"key": "value"}),
        ('{"a": True, "b": False}', {"a": True, "b": False}),
    ],
)
def test_llm_response_as_json(in_str, expected):
    """Test the `llm_response_as_json` function"""

    assert llm_response_as_json(in_str) == expected


@pytest.mark.parametrize(
    "text_chunks,n,expected",
    [
        (
            [
                (
                    "Some text. Some overlap. More text. More text that "
                    "shouldn't be touched. Some overlap."
                ),
                "Some overlap. More text.",
                "Some non-overlapping text.",
            ],
            12,
            (
                "Some text. Some overlap. More text. More text that "
                "shouldn't be touched. Some overlap. More text.\nSome "
                "non-overlapping text."
            ),
        ),
        ([], 300, ""),
        (["single chunk"], 300, "single chunk"),
        (["", None, "valid text"], 300, "valid text"),
    ],
)
def test_merge_overlapping_texts(text_chunks, n, expected):
    """Test the `merge_overlapping_texts` function"""

    assert merge_overlapping_texts(text_chunks, n) == expected


@pytest.mark.parametrize(
    "doc_attrs,expected",
    [
        ({"date": (2023, 5, 15)}, 2023),
        ({"date": (2020, 1, 1)}, 2020),
        ({"date": (None, 5, 15)}, None),
        ({"date": (0, 5, 15)}, None),
        ({"date": (-1, 5, 15)}, None),
        ({}, None),
        ({"other_key": "value"}, None),
    ],
)
def test_extract_year_from_doc_attrs(doc_attrs, expected):
    """Test the `extract_year_from_doc_attrs` function"""

    assert extract_year_from_doc_attrs(doc_attrs) == expected


def test_num_ordinances_dataframe_empty():
    """Test `num_ordinances_dataframe` with empty DataFrame"""

    data = pd.DataFrame()
    assert num_ordinances_dataframe(data) == 0


def test_num_ordinances_dataframe_with_values():
    """Test `num_ordinances_dataframe` with valid data"""

    data = pd.DataFrame(
        {
            "feature": ["setback", "height", "noise"],
            "value": [100, 200, None],
            "summary": ["test", None, "test"],
        }
    )
    assert num_ordinances_dataframe(data) == 3


def test_num_ordinances_dataframe_with_exclude():
    """Test `num_ordinances_dataframe` with excluded features"""

    data = pd.DataFrame(
        {
            "feature": ["setback", "HEIGHT", "noise"],
            "value": [100, 200, 300],
            "summary": ["test", "test", "test"],
        }
    )
    assert num_ordinances_dataframe(data, exclude_features=["height"]) == 2


def test_ordinances_bool_index_none():
    """Test `ordinances_bool_index` with None data"""

    result = ordinances_bool_index(None)
    assert isinstance(result, np.ndarray)
    assert len(result) == 0
    assert result.dtype == bool


def test_ordinances_bool_index_empty():
    """Test `ordinances_bool_index` with empty DataFrame"""

    data = pd.DataFrame()
    result = ordinances_bool_index(data)
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


def test_ordinances_bool_index_no_check_cols():
    """Test `ordinances_bool_index` with DataFrame missing required columns"""

    data = pd.DataFrame({"feature": ["setback", "height"]})
    result = ordinances_bool_index(data)
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


def test_ordinances_bool_index_with_values():
    """Test `ordinances_bool_index` with valid data"""

    data = pd.DataFrame(
        {
            "feature": ["setback", "height", "noise", "extra"],
            "value": [100, None, None, 400],
            "summary": [None, "test", None, "test"],
        }
    )
    result = ordinances_bool_index(data)
    expected = np.array([True, True, False, True])
    np.testing.assert_array_equal(result, expected)


def test_ordinances_bool_index_value_only():
    """Test `ordinances_bool_index` with only value column"""

    data = pd.DataFrame(
        {"feature": ["setback", "height", "noise"], "value": [100, None, 300]}
    )
    result = ordinances_bool_index(data)
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(result, expected)


def test_convert_paths_to_strings_all_structures():
    """Test `convert_paths_to_strings` across nested containers"""

    input_obj = {
        Path("path_key"): {
            "list": [
                Path("inner_list_item"),
                {Path("dict_key"): Path("dict_value")},
            ],
            "tuple": (Path("inner_tuple_item"), "preserve"),
            "set": {Path("inner_set_item"), "inner_literal"},
        },
        "list": [Path("top_list_item"), (Path("tuple_in_list"),)],
        "tuple": (Path("top_tuple_item"), {Path("tuple_set_item")}),
        "set": {
            Path("top_set_item"),
            ("nested_tuple", Path("nested_tuple_path")),
        },
        "value": "literal",
        "path_value": Path("top_value_path"),
    }

    result = convert_paths_to_strings(input_obj)

    expected = {
        "path_key": {
            "list": [
                "inner_list_item",
                {"dict_key": "dict_value"},
            ],
            "tuple": ("inner_tuple_item", "preserve"),
            "set": {"inner_set_item", "inner_literal"},
        },
        "list": ["top_list_item", ("tuple_in_list",)],
        "tuple": ("top_tuple_item", {"tuple_set_item"}),
        "set": {"top_set_item", ("nested_tuple", "nested_tuple_path")},
        "value": "literal",
        "path_value": "top_value_path",
    }

    assert result == expected


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
