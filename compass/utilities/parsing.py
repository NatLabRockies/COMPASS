"""COMPASS ordinance parsing utilities"""

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
from elm.web.document import PDFDocument


logger = logging.getLogger(__name__)
_ORD_CHECK_COLS = ["value", "summary", "ordinance_text"]


def is_pdf_doc(doc):
    """Determine whether a document is a PDF based on type or attributes

    This function first checks if the document is an instance of
    PDFDocument. If not, it looks for a "doc_type" attribute in the
    document's attributes and checks if it is a string that
    case-insensitively matches "pdf". If neither condition is met, the
    function returns ``False``.

    Parameters
    ----------
    doc : elm.web.document.Document
        Document instance to check for PDF characteristics. The function
        first checks if the document is an instance of PDFDocument. If
        not, it looks for a "doc_type" attribute in the document's
        attributes and checks if it is a string that case-insensitively
        matches "pdf". If neither condition is met, the function returns
        ``False``.

    Returns
    -------
    bool
        ``True`` when a document represents a PDF file, ``False``
        otherwise.
    """
    doc_type = doc.attrs.get("doc_type") or ""
    return isinstance(doc, PDFDocument) or doc_type.casefold() == "pdf"


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
    candidates = [content]
    candidates.extend(
        reversed(
            [
                candidate.strip()
                for candidate in re.findall(
                    r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL
                )
                if candidate.strip()
            ]
        )
    )

    decode_error = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.decoder.JSONDecodeError as error:
            decode_error = error

        normalized = _normalize_python_bool_literals(candidate)
        if normalized != candidate:
            try:
                return json.loads(normalized)
            except json.decoder.JSONDecodeError:
                pass

    embedded = _extract_embedded_json(content)
    if embedded is not None:
        logger.debug(
            "LLM response contained leading/trailing text around the "
            "JSON payload. Recovered JSON from an embedded object or "
            "array instead of the full response."
        )
        return embedded

    logger.error(
        "LLM returned improperly formatted JSON. "
        "This is likely due to the completion running out of tokens. "
        "Setting a higher token limit may fix this error. "
        "Also ensure you are requesting JSON output in your prompt. "
        "JSON returned:\n%s",
        content,
        exc_info=decode_error,
    )
    return {}


def _extract_embedded_json(content):
    """Recover a JSON object or array embedded in surrounding text."""
    decoder = json.JSONDecoder()
    for index in _json_start_indexes(content):
        candidate = content[index:]
        if candidate.startswith("[") and candidate[1:2].isdigit():
            continue
        for candidate_text in (
            candidate,
            _normalize_python_bool_literals(candidate),
        ):
            try:
                value, _ = decoder.raw_decode(candidate_text)
            except json.decoder.JSONDecodeError:
                continue
            if (
                isinstance(value, list)
                and len(value) == 1
                and isinstance(value[0], int | float)
            ):
                continue
            return value
    return None


def _json_start_indexes(content):
    """Yield likely JSON opener positions in source order."""
    return [index for index, char in enumerate(content) if char in "[{"]


def _normalize_python_bool_literals(content):
    """Lowercase unquoted Python boolean literals for JSON parsing"""
    out = []
    index = 0
    in_string = False
    escaped = False
    while index < len(content):
        char = content[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue

        literal, replacement = _matching_bool_literal(content, index)
        if literal is not None:
            out.append(replacement)
            index += len(literal)
            continue

        out.append(char)
        index += 1
    return "".join(out)


def _matching_bool_literal(content, index):
    """Return bool literal replacement when the token is isolated."""
    for literal, replacement in {"True": "true", "False": "false"}.items():
        if not content.startswith(literal, index):
            continue
        if _is_bool_literal_boundary(content, index, literal):
            return literal, replacement
    return None, None


def _is_bool_literal_boundary(content, index, literal):
    """Determine whether a boolean literal token is standalone."""
    prev_char = content[index - 1] if index > 0 else ""
    next_index = index + len(literal)
    next_char = content[next_index] if next_index < len(content) else ""
    prev_is_word = prev_char.isalnum() or prev_char == "_"
    next_is_word = next_char.isalnum() or next_char == "_"
    return not (prev_is_word or next_is_word)


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


def extract_year_from_doc_attrs(doc_attrs):
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
    >>> extract_year_from_doc_attrs({"date": (2024, 5, 17)})
    2024
    >>> extract_year_from_doc_attrs({"date": (None, None, None)})
    None
    """
    year, *__ = doc_attrs.get("date") or (None, None, None)
    return year if year is not None and year > 0 else None


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

    return int(ordinances_bool_index(data).sum())


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


def raw_pages_from_doc(
    doc,
    text_splitter=None,
    percent_raw_pages_to_keep=25,
    max_raw_pages=18,
    num_end_pages_to_keep=2,
):
    """[NOT PUBLIC API] Get raw pages from an input doc"""
    # Do NOT use `is_pdf_doc` here because MDDocuments could have
    # "doc_type" == "pdf" and be treated as a single page doc
    if isinstance(doc, PDFDocument) and hasattr(doc, "raw_pages"):
        return _raw_pages_from_pdf_doc(
            doc,
            text_splitter,
            percent_raw_pages_to_keep,
            max_raw_pages,
            num_end_pages_to_keep,
        )

    if text_splitter is None:
        logger.debug(
            "Cannot split out raw pages for document from %s because no "
            "text splitter provided",
            doc.attrs.get("source", "unknown source"),
        )
        return [doc.text]

    text = "\n\n".join(doc.pages)
    if not text:
        return []

    pages = text_splitter.split_text(text)
    raw_pages = _down_select_pages(
        pages, percent_raw_pages_to_keep, max_raw_pages, num_end_pages_to_keep
    )

    logger.debug(
        "Document from %s has %d raw %s after splitting and trimming",
        doc.attrs.get("source", "unknown source"),
        len(raw_pages),
        "page" if len(raw_pages) == 1 else "pages",
    )
    return raw_pages


def _raw_pages_from_pdf_doc(
    doc,
    text_splitter,
    percent_raw_pages_to_keep,
    max_raw_pages,
    num_end_pages_to_keep,
):
    """Get raw pages from an input PDF doc"""
    raw_pages = doc.raw_pages
    # failsafe check
    if text_splitter is not None and len(raw_pages) == 1:
        raw_pages = text_splitter.split_text(raw_pages[0])
        raw_pages = _down_select_pages(
            raw_pages,
            percent_raw_pages_to_keep,
            max_raw_pages,
            num_end_pages_to_keep,
        )
        logger.debug(
            "PDF Document from %s had 1 raw page; "
            "has %d raw %s after splitting",
            doc.attrs.get("source", "unknown source"),
            len(raw_pages),
            "page" if len(raw_pages) == 1 else "pages",
        )
        return raw_pages

    logger.debug(
        "PDF Document from %s has %d raw %s",
        doc.attrs.get("source", "unknown source"),
        len(raw_pages),
        "page" if len(raw_pages) == 1 else "pages",
    )
    return raw_pages


def _down_select_pages(
    pages, percent_raw_pages_to_keep, max_raw_pages, num_end_pages_to_keep
):
    """Down-select pages based on percentage and max limits"""
    num_to_keep = percent_raw_pages_to_keep / 100 * len(pages)
    num_raw_pages_to_keep = min(max_raw_pages, max(1, int(num_to_keep)))

    neg_num_extra_pages = num_raw_pages_to_keep - len(pages)
    neg_num_last_pages = max(-num_end_pages_to_keep, neg_num_extra_pages)
    last_page_index = min(0, neg_num_last_pages)

    raw_pages = pages[:num_raw_pages_to_keep]
    if last_page_index:
        raw_pages += pages[last_page_index:]

    return raw_pages


# complexipy: ignore
def convert_paths_to_strings(obj):
    """[NOT PUBLIC API] Convert all Path instances to strings"""
    logger.trace("Converting paths to strings in object: %s", obj)
    if isinstance(obj, Path):
        out = os.fspath(obj)
        if not obj.is_absolute():
            out = os.path.join(".", out)  # ruff:ignore[os-path-join] PTH118
        return out
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
