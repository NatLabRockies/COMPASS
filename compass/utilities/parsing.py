"""COMPASS Ordinances parsing utilities."""

import json
import logging


logger = logging.getLogger(__name__)
_ORD_CHECK_COLS = ["value", "adder", "min_dist", "max_dist", "summary"]


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


def num_ordinances_in_doc(doc):
    """Count number of ordinances found in document

    Parameters
    ----------
    doc : elm.web.document.Document
        Document potentially containing ordinances for a jurisdiction.
        If no ordinance values are found, this function returns ``0``.

    Returns
    -------
    int
        Number of unique ordinance values extracted from this document.
    """
    if doc is None:
        return 0

    if "ordinance_values" not in doc.attrs:
        return 0

    return num_ordinances_dataframe(doc.attrs["ordinance_values"])


def num_ordinances_dataframe(data):
    """Count number of ordinances found in DataFrame

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame potentially containing ordinances for a jurisdiction.
        If no ordinance values are found, this function returns ``0``.

    Returns
    -------
    int
        Number of unique ordinance values extracted from this DataFrame.
    """
    if data is None:
        return 0

    if data.empty:
        return 0

    check_cols = [col for col in _ORD_CHECK_COLS if col in data]
    if not check_cols:
        return 0

    found_features = (~data[check_cols].isna()).to_numpy().sum(axis=1)
    return (found_features > 0).sum()
