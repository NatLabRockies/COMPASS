"""COMPASS ordinance parsing utilities"""

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
        LLM response that may contain markdown triple backticks.

    Returns
    -------
    str
        Response stripped of all leading and trailing backtick markers.
    """
    content = content.lstrip().rstrip()
    return content.removeprefix("```").lstrip("\n").removesuffix("```")


def llm_response_as_json(content):
    """Parse a raw LLM response into JSON-compatible data

    Parameters
    ----------
    content : str
        Response text expected to contain a JSON object, possibly with
        Markdown fences or Python boolean literals.

    Returns
    -------
    dict
        Parsed JSON structure. When parsing fails, the function returns
        an empty dictionary.

    Notes
    -----
    The parser strips Markdown code fences, coerces Python-style
    booleans to lowercase JSON literals, and logs the raw response on
    decode failure. The logging includes guidance for increasing token
    limits or updating prompts.
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
    """Merge text chunks while trimming overlapping boundaries

    Overlap detection compares at most ``n`` characters at each
    boundary but never more than half the length of the accumulated
    output. Chunks that do not overlap are concatenated with a newline
    separator.

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
        Merged text assembled from the non-overlapping portions.
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
    """Extract the ordinance year stored in document attributes

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
        Parsed ordinance year or ``None`` when unavailable or invalid.

    Examples
    --------
    >>> extract_ord_year_from_doc_attrs({"date": (2024, 5, 17)})
    2024
    >>> extract_ord_year_from_doc_attrs({"date": (None, None, None)})
    None
    """
    year = doc_attrs.get("date", (None, None, None))[0]
    return year if year is not None and year > 0 else None


def num_ordinances_in_doc(doc, exclude_features=None):
    """Count the number of ordinance entries on a document

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
        Number of ordinance rows represented in ``doc``.
    """
    if doc is None or doc.attrs.get("ordinance_values") is None:
        return 0

    return num_ordinances_dataframe(
        doc.attrs["ordinance_values"], exclude_features=exclude_features
    )


def num_ordinances_dataframe(data, exclude_features=None):
    """Count ordinance rows contained in a DataFrame

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
        Count of rows meeting the ordinance criteria.

    Raises
    ------
    KeyError
        If the input DataFrame lacks the ``feature`` column when
        ``exclude_features`` is provided.
    """
    if exclude_features:
        mask = ~data["feature"].str.casefold().isin(exclude_features)
        data = data[mask].copy()

    return ordinances_bool_index(data).sum()


def ordinances_bool_index(data):
    """Compute a boolean mask indicating ordinance rows

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame potentially containing ordinances for a jurisdiction.
        If no ordinance values are found, this function returns ``0``.

    Returns
    -------
    numpy.ndarray
        Boolean mask identifying rows that contain ordinance values.
    """
    if data is None or data.empty:
        return np.array([], dtype=bool)

    check_cols = [col for col in _ORD_CHECK_COLS if col in data]
    if not check_cols:
        return np.array([], dtype=bool)

    found_features = (~data[check_cols].isna()).to_numpy().sum(axis=1)
    return found_features > 0


def load_config(config_fp):
    """Load configuration data from JSON or JSON5 sources

    Parameters
    ----------
    config_fp : path-like
        Path to config file to open and load.

    Returns
    -------
    dict
        Parsed configuration object.

    Raises
    ------
    COMPASSValueError
        If the file path does not exist or the extension is not
        ``.json`` or ``.json5``.

    Notes
    -----
    JSON5 enables comments and trailing commas, among other
    quality-of-life improvements over vanilla JSON.
    """
    config_fp = Path(config_fp)

    if not config_fp.exists():
        msg = f"Config file does not exist: {config_fp}"
        raise COMPASSValueError(msg)

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


def convert_paths_to_strings(obj):
    """[NOT PUBLIC API] Convert all Path instances to strings"""
    logger.trace("Converting paths to strings in object: %s", obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {
            convert_paths_to_strings(key): convert_paths_to_strings(value)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [convert_paths_to_strings(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(convert_paths_to_strings(item) for item in obj)
    if isinstance(obj, set):
        return {convert_paths_to_strings(item) for item in obj}
    return obj
