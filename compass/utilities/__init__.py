"""Ordinance utilities"""

from .counties import load_all_county_info, load_counties_from_fp
from .parsing import (
    extract_ord_year_from_doc_attrs,
    llm_response_as_json,
    merge_overlapping_texts,
    num_ordinances_in_doc,
    num_ordinances_dataframe,
)


RTS_SEPARATORS = [
    r"Chapter \d+",
    r"Section \d+",
    r"Article \d+",
    "CHAPTER ",
    "SECTION ",
    "Chapter ",
    "Section ",
    r"\n[\s]*\d+\.\d+ [A-Z]",  # match "\n\t  123.24 A"
    r"\n[\s]*\d+\.\d+\.\d+ ",  # match "\n\t 123.24.250 "
    r"\n[\s]*\d+\.\d+\.",  # match "\n\t 123.24."
    r"\n[\s]*\d+\.\d+\.\d+\.",  # match "\n\t 123.24.250."
    "Setbacks",
    "\r\n\r\n",
    "\r\n",
    "\n\n",
    "\n",
    "section ",
    "chapter ",
    " ",
    "",
]


LLM_COST_REGISTRY = {
    "o1": {"prompt": 15, "response": 60},
    "o3-mini": {"prompt": 1.1, "response": 4.4},
    "gpt-4.5": {"prompt": 75, "response": 150},
    "gpt-4o": {"prompt": 2.5, "response": 10},
    "gpt-4o-mini": {"prompt": 0.15, "response": 0.6},
    "wetosa-gpt-4o-mini": {"prompt": 0.15, "response": 0.6},
}
"""LLM Costs registry

The registry maps model names to a dictionary that contains the cost
(in $/million tokens) for both prompt and response tokens.
"""
