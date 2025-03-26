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
    "filetype:pdf {location} wind energy conversion system ordinances",
    "wind energy conversion system ordinances {location}",
    "{location} wind WECS ordinance",
    "Where can I find the legal text for commercial wind energy "
    "conversion system zoning ordinances in {location}?",
    "What is the specific legal information regarding zoning "
    "ordinances for commercial wind energy conversion systems in {location}?",
]
