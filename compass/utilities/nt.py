"""COMPASS namedtuple data classes"""

from collections import namedtuple

ProcessKwargs = namedtuple(
    "ProcessKwargs",
    [
        "known_local_docs",
        "known_doc_urls",
        "file_loader_kwargs",
        "td_kwargs",
        "tpe_kwargs",
        "ppe_kwargs",
        "max_num_concurrent_jurisdictions",
    ],
    defaults=[None, None, None, None, 25],
)
ProcessKwargs.__doc__ = """Execution options passed to `compass process`

Parameters
----------
known_local_docs : list of path-like, optional
    Local ordinance files to seed the run. ``None`` disables the seed.
    By default, ``None``.
known_doc_urls : list of str, optional
    Known ordinance URLs to prioritize during retrieval.
    By default, ``None``.
file_loader_kwargs : dict, optional
    Keyword arguments forwarded to the document loader implementation.
    By default, ``None``.
td_kwargs : dict, optional
    Additional configuration for top-level document discovery logic.
    By default, ``None``.
tpe_kwargs : dict, optional
    Parameters controlling text parsing and extraction.
    By default, ``None``.
ppe_kwargs : dict, optional
    Parameters controlling permitted-use parsing and extraction.
    By default, ``None``.
max_num_concurrent_jurisdictions : int, default=25
    Maximum number of jurisdictions processed simultaneously.
    By default, ``25``.
"""
