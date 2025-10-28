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
