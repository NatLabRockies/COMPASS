"""Solar ordinance extraction utilities"""

from .ordinance import (
    SolarOrdinanceTextExtractor,
    SolarOrdinanceValidator,
)
from .parse import StructuredSolarOrdinanceParser


SOLAR_QUESTION_TEMPLATES = [
    "filetype:pdf {location} solar energy conversion system zoning ordinances",
    "solar energy conversion system zoning ordinances {location}",
    "{location} solar energy farm zoning ordinance",
    "Where can I find the legal text for commercial solar energy "
    "conversion system zoning ordinances in {location}?",
    "What is the specific legal information regarding zoning "
    "ordinances for commercial solar energy conversion systems in {location}?",
]
