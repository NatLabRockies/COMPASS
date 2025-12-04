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

TechSpec = namedtuple(
    "TechSpec",
    [
        "name",
        "questions",
        "heuristic",
        "ordinance_text_collector",
        "ordinance_text_extractor",
        "permitted_use_text_collector",
        "permitted_use_text_extractor",
        "structured_ordinance_parser",
        "structured_permitted_use_parser",
        "website_url_keyword_points",
    ],
)
TechSpec.__doc__ = """Bundle extraction configuration for a technology

Parameters
----------
name : str
    Display name for the technology (e.g., ``"solar"``).
questions : dict
    Prompt templates or question sets used during extraction.
heuristic : callable
    Function implementing heuristic filters prior to LLM invocation.
ordinance_text_collector : callable
    Callable that gathers candidate ordinance text spans.
ordinance_text_extractor : callable
    Callable that extracts relevant ordinance snippets.
permitted_use_text_collector : callable
    Callable that gathers candidate permitted-use text spans.
permitted_use_text_extractor : callable
    Callable that extracts permitted-use content.
structured_ordinance_parser : callable
    Callable that transforms ordinance text into structured values.
structured_permitted_use_parser : callable
    Callable that transforms permitted-use text into structured values.
website_url_keyword_points : dict or None
    Weightings for scoring website URLs during search.
"""
