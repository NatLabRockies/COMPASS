"""Post processing functions for one-shot plugins"""

import inspect
import re
from pathlib import Path


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


def normalize_data_center_types(db):
    """Leave generic types blank and shorten accessory type labels"""
    db["data_center_type"] = db["data_center_type"].map(
        _normalize_data_center_type
    )
    return db


def _normalize_data_center_type(value):
    if not isinstance(value, str):
        return value

    value = " ".join(value.split())
    if value.casefold() in {
        "",
        "data center",
        "data centers",
        "datacenter",
        "datacenters",
        "all",
        "all types",
        "all data centers",
        "all data center types",
    }:
        return None

    if re.fullmatch(
        r"accessory data center(?:\s*\(.*\))?", value, re.IGNORECASE
    ):
        return "Accessory Data Center"

    return value


POST_PROCESSING_REGISTRY = {
    name: func
    for name, func in globals().items()
    if inspect.isfunction(func)
    and func.__module__ == __name__
    and not name.startswith("_")
}
"""[NOT PUBLIC API] Post-processing step registry"""
