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
