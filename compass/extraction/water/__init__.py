"""Water ordinance extraction utilities"""

from .parse import StructuredWaterParser
from .ordinance import (
    WaterRightsHeuristic,
    WaterRightsTextCollector,
    WaterRightsTextExtractor,
)
from .processing import (
    build_corpus,
    extract_water_rights_ordinance_values,
    label_docs_no_legal_check,
    write_water_rights_data_to_disk,
)


WATER_RIGHTS_QUESTION_TEMPLATES = [
    "{jurisdiction} rules",
    "{jurisdiction} management plan",
    "{jurisdiction} well permits",
    "{jurisdiction} well permit requirements",
    "requirements to drill a water well in {jurisdiction}",
]

BEST_WATER_RIGHTS_ORDINANCE_WEBSITE_URL_KEYWORDS = {
    "pdf": 92160,
    "water": 46080,
    "rights": 23040,
    "zoning": 11520,
    "ordinance": 5760,
    r"renewable%20energy": 1440,
    r"renewable+energy": 1440,
    "renewable energy": 1440,
    "planning": 720,
    "plan": 360,
    "government": 180,
    "code": 60,
    "area": 60,
    r"land%20development": 15,
    r"land+development": 15,
    "land development": 15,
    "land": 3,
    "environment": 3,
    "energy": 3,
    "renewable": 3,
    "municipal": 1,
    "department": 1,
}
