"""COMPASS Ordinances parsing utilities."""

import json
import logging
from pathlib import Path

import pyjson5
import numpy as np

from compass.exceptions import COMPASSValueError

logger = logging.getLogger(__name__)
_ORD_CHECK_COLS = ["value", "summary"]


def clean_backticks_from_llm_response(content):
    """Remove markdown-style backticks from an LLM response

    Parameters
    ----------
    content : str
        LLM response that may or may not contain markdown-style triple
        backticks.

    Returns
    -------
    str
        LLM response stripped of the markdown-style backticks
    """
    content = content.lstrip().rstrip()
    return content.removeprefix("```").lstrip("\n").removesuffix("```")


def llm_response_as_json(content):
    """LLM response to JSON

    Parameters
    ----------
    content : str
        LLM response that contains a string representation of
        a JSON file.

    Returns
    -------
    dict
        Response parsed into dictionary. This dictionary will be empty
        if the response cannot be parsed by JSON.
    """
    content = clean_backticks_from_llm_response(content)
    content = content.removeprefix("json").lstrip("\n")
    content = content.replace("True", "true").replace("False", "false")
    try:
        content = json.loads(content)
    except json.decoder.JSONDecodeError:
        logger.exception(
            "LLM returned improperly formatted JSON. "
            "This is likely due to the completion running out of tokens. "
            "Setting a higher token limit may fix this error. "
            "Also ensure you are requesting JSON output in your prompt. "
            "JSON returned:\n%s",
            content,
        )
        content = {}
    return content


def merge_overlapping_texts(text_chunks, n=300):
    """Merge chunks of text by removing any overlap.

    Parameters
    ----------
    text_chunks : iterable of str
        Iterable containing text chunks which may or may not contain
        consecutive overlapping portions.
    n : int, optional
        Number of characters to check at the beginning of each message
        for overlap with the previous message. Will always be reduced to
        be less than or equal to half of the length of the previous
        chunk. By default, ``300``.

    Returns
    -------
    str
        Merged text.
    """
    text_chunks = list(filter(None, text_chunks))
    if not text_chunks:
        return ""

    out_text = text_chunks[0]
    for next_text in text_chunks[1:]:
        half_chunk_len = len(out_text) // 2
        check_len = min(n, half_chunk_len)
        next_chunks_start_ind = out_text[half_chunk_len:].find(
            next_text[:check_len]
        )
        if next_chunks_start_ind == -1:
            out_text = f"{out_text}\n{next_text}"
            continue
        next_chunks_start_ind += half_chunk_len
        out_text = "".join([out_text[:next_chunks_start_ind], next_text])
    return out_text


def extract_ord_year_from_doc_attrs(doc_attrs):
    """Extract year corresponding to the ordinance from doc instance

    Parameters
    ----------
    doc_attrs : dict
        Document meta information about the jurisdiction.
        Must have a "date" key in the attrs that is a tuple
        corresponding to the (year, month, day) of the ordinance to
        extract year successfully. If this key is missing, this function
        returns ``None``.

    Returns
    -------
    int or None
        Parsed year for ordinance (int) or ``None`` if it wasn't found
        in the document's attrs.
    """
    year = doc_attrs.get("date", (None, None, None))[0]
    return year if year is not None and year > 0 else None


def num_ordinances_in_doc(doc, exclude_features=None):
    """Count number of ordinances found in document

    Parameters
    ----------
    doc : elm.web.document.BaseDocument
        Document potentially containing ordinances for a jurisdiction.
        If no ordinance values are found, this function returns ``0``.
    exclude_features : iterable of str, optional
        Optional features to exclude from ordinance count.
        By default, ``None``.

    Returns
    -------
    int
        Number of unique ordinance values extracted from this document.
    """
    if doc is None or doc.attrs.get("ordinance_values") is None:
        return 0

    return num_ordinances_dataframe(
        doc.attrs["ordinance_values"], exclude_features=exclude_features
    )


def num_ordinances_dataframe(data, exclude_features=None):
    """Count number of ordinances found in DataFrame

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame potentially containing ordinances for a jurisdiction.
        If no ordinance values are found, this function returns ``0``.
    exclude_features : iterable of str, optional
        Optional features to exclude from ordinance count.
        By default, ``None``.

    Returns
    -------
    int
        Number of unique ordinance values extracted from this DataFrame.
    """
    if exclude_features:
        mask = ~data["feature"].str.casefold().isin(exclude_features)
        data = data[mask].copy()

    return ordinances_bool_index(data).sum()


def ordinances_bool_index(data):
    """Array of bools indicating rows containing ordinances in DataFrame

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame potentially containing ordinances for a jurisdiction.
        If no ordinance values are found, this function returns ``0``.

    Returns
    -------
    array-like
        Array of bools indicating rows containing ordinances in
        DataFrame.
    """
    if data is None or data.empty:
        return np.array([], dtype=bool)

    check_cols = [col for col in _ORD_CHECK_COLS if col in data]
    if not check_cols:
        return np.array([], dtype=bool)

    found_features = (~data[check_cols].isna()).to_numpy().sum(axis=1)
    return found_features > 0


def load_config(config_fp):
    """Load a JSON or JSON5 config file

    Parameters
    ----------
    config_fp : path-like
        Path to config file to open and load.

    Returns
    -------
    dict
        Dictionary containing the config file contents.

    Raises
    ------
    COMPASSValueError
        If the config file does not end with `.json` or `.json5`.
    """
    config_fp = Path(config_fp)

    if config_fp.suffix == ".json5":
        with config_fp.open(encoding="utf-8") as fh:
            return pyjson5.decode_io(fh)

    if config_fp.suffix == ".json":
        with config_fp.open(encoding="utf-8") as fh:
            return json.load(fh)

    msg = (
        "Got unknown config file extension: "
        f"{config_fp.suffix}. Supported extensions are .json5 and .json."
    )
    raise COMPASSValueError(msg)
