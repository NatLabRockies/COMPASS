"""Wind ordinance extraction utilities"""

from .ordinance import (
    WindOrdinanceTextExtractor,
    WindOrdinanceValidator,
)
from .parse import StructuredWindOrdinanceParser


WIND_QUESTION_TEMPLATES = [
    "filetype:pdf {location} wind energy conversion system zoning ordinances",
    "wind energy conversion system zoning ordinances {location}",
    "{location} wind WECS zoning ordinance",
    "Where can I find the legal text for commercial wind energy "
    "conversion system zoning ordinances in {location}?",
    "What is the specific legal information regarding zoning "
    "ordinances for commercial wind energy conversion systems in {location}?",
]
