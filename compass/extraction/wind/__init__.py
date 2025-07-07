"""Wind ordinance extraction utilities"""

from .ordinance import (
    WindHeuristic,
    WindOrdinanceTextCollector,
    WindOrdinanceTextExtractor,
    WindPermittedUseDistrictsTextCollector,
    WindPermittedUseDistrictsTextExtractor,
)
from .parse import (
    StructuredWindOrdinanceParser,
    StructuredWindPermittedUseDistrictsParser,
)


WIND_QUESTION_TEMPLATES = [
    "filetype:pdf {jurisdiction} wind energy conversion system ordinances",
    "wind energy conversion system ordinances {jurisdiction}",
    "{jurisdiction} wind WECS ordinance",
    "Where can I find the legal text for commercial wind energy "
    "conversion system zoning ordinances in {jurisdiction}?",
    "What is the specific legal information regarding zoning "
    "ordinances for commercial wind energy conversion systems in "
    "{jurisdiction}?",
]

BEST_WIND_ORDINANCE_WEBSITE_URL_KEYWORDS = {
    "pdf": 92160,
    "wecs": 46080,
    "wind": 23040,
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
