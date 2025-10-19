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
    "pdf": 92160,
    "secs": 46080,
    "solar": 23040,
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
    # TODO: add board???
}
