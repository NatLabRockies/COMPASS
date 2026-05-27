"""Ordinance utilities"""

from .base import Directories, title_preserving_caps
from .costs import (
    LLM_COST_REGISTRY,
    cost_for_model,
    compute_cost_from_totals,
    compute_total_cost_from_usage,
)
from .finalize import (
    compile_run_summary_message,
    doc_infos_to_db,
    save_db,
    save_run_meta,
)
from .jurisdictions import (
    load_all_jurisdiction_info,
    load_jurisdictions_from_fp,
)
from .parsing import (
    extract_year_from_doc_attrs,
    llm_response_as_json,
    merge_overlapping_texts,
    num_ordinances_dataframe,
    ordinances_bool_index,
)
from .nt import ProcessKwargs


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
