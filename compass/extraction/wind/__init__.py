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
    "pdf": 4608,
    "wecs": 2304,
    "wind": 1152,
    "zoning": 576,
    "ordinance": 288,
    "planning": 144,
    "plan": 72,
    "government": 36,
    "code": 12,
    "area": 12,
    "land development": 6,
    "land": 3,
    "municipal": 1,
    "department": 1,
}
