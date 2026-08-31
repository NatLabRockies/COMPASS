"""Post processing functions for one-shot plugins"""

import inspect
from pathlib import Path

from compass.utilities.finalize import (
    MAX_ORDINANCE_TEXT_CHARS,  # ruff: ignore[unused-import]
    trim_ordinance_text,
)


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
# defined in compass.utilities.finalize (where it is also applied
# unconditionally), so it is registered here by hand
POST_PROCESSING_REGISTRY["trim_ordinance_text"] = trim_ordinance_text
"""[NOT PUBLIC API] Post-processing step registry"""
