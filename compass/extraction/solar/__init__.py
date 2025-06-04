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
