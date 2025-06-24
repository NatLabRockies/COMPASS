"""Solar ordinance extraction utilities"""

from .ordinance import (
    SolarHeuristic,
    SolarOrdinanceTextCollector,
    SolarOrdinanceTextExtractor,
    SolarPermittedUseDistrictsTextCollector,
    SolarPermittedUseDistrictsTextExtractor,
)
from .parse import (
    StructuredSolarOrdinanceParser,
    StructuredSolarPermittedUseDistrictsParser,
)


SOLAR_QUESTION_TEMPLATES = [
    "filetype:pdf {jurisdiction} solar energy conversion system ordinances",
    "solar energy conversion system ordinances {jurisdiction}",
    "{jurisdiction} solar energy farm ordinance",
    "Where can I find the legal text for commercial solar energy "
    "conversion system zoning ordinances in {jurisdiction}?",
    "What is the specific legal information regarding zoning "
    "ordinances for commercial solar energy conversion systems in "
    "{jurisdiction}?",
]

BEST_SOLAR_ORDINANCE_WEBSITE_URL_KEYWORDS = {
    "pdf": 4608,
    "secs": 2304,
    "solar": 1152,
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
