"""Post processing functions for one-shot plugins"""

import inspect
from pathlib import Path


MAX_ORDINANCE_TEXT_CHARS = 5000
"""int: Maximum number of characters kept in the ``ordinance_text`` column"""


def trim_ordinance_text(db):
    """Trim the ``ordinance_text`` column to a maximum character count

    LLMs occasionally return a very long excerpt for ``ordinance_text``,
    which makes the output CSV unwieldy. Any excerpt longer than
    :data:`MAX_ORDINANCE_TEXT_CHARS` is cut back to the last whole word
    that fits and marked with a trailing ellipsis, matching the ellipsis
    convention already used for elided text within an excerpt.

    Parameters
    ----------
    db : pandas.DataFrame
        The database containing extraction results, which may include an
        ``"ordinance_text"`` column.

    Returns
    -------
    pandas.DataFrame
        The updated database, with any over-long ``ordinance_text``
        entries trimmed. Databases without that column are returned
        unchanged.
    """
    if db.empty or "ordinance_text" not in db.columns:
        return db

    db["ordinance_text"] = db["ordinance_text"].apply(_trim_excerpt)
    return db


def _trim_excerpt(text):
    """Cut an excerpt to the last whole word within the char limit"""
    if not isinstance(text, str) or len(text) <= MAX_ORDINANCE_TEXT_CHARS:
        return text

    kept = text[:MAX_ORDINANCE_TEXT_CHARS].rsplit(" ", 1)[0].rstrip()
    return f"{kept} ..."


def add_document_name(db):
    """Add a document_name column to the database

    The document_name is derived from the source path, if available.

    Parameters
    ----------
    db : pandas.DataFrame
        The database containing extraction results, which may include a
        'source' column.

    Returns
    -------
    pandas.DataFrame
        The updated database with an added 'document_name' column, if
        applicable.
    """
    if not db.empty:
        db["document_name"] = db["source"].apply(
            lambda src: (
                Path(src).name if isinstance(src, str) and src else None
            )
        )
    return db


POST_PROCESSING_REGISTRY = {
    name: func
    for name, func in globals().items()
    if inspect.isfunction(func)
    and func.__module__ == __name__
    and not name.startswith("_")
}
"""[NOT PUBLIC API] Post-processing step registry"""
