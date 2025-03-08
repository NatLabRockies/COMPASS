"""Ordinance utilities"""

from .counties import load_all_county_info, load_counties_from_fp
from .parsing import (
    extract_ord_year_from_doc_attrs,
    llm_response_as_json,
    merge_overlapping_texts,
    num_ordinances_in_doc,
)


RTS_SEPARATORS = [
    r"\d+\.\d+ ",  # match "123.24 "
    r"\d+\.\d+\.\d+ ",  # match "123.24.250 "
    r"\d+\.\d+\.",  # match "123.24."
    r"\d+\.\d+\.\d+\.",  # match "123.24.250."
    r"Section \d+",
    "CHAPTER ",
    "SECTION ",
    "Chapter ",
    "Section ",
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
