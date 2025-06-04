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
