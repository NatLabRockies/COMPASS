"""Wind ordinance extraction utilities"""

from .ordinance import (
    possibly_mentions_wind,
    WindOrdinanceTextExtractor,
    WindOrdinanceValidator,
)
from .parse import StructuredWindOrdinanceParser


WIND_QUESTION_TEMPLATES = [
    '0. "wind energy conversion system zoning ordinances {location}"',
    '1. "{location} wind WECS zoning ordinance"',
    '2. "Where can I find the legal text for commercial wind energy '
    'conversion system zoning ordinances in {location}?"',
    '3. "What is the specific legal information regarding zoning '
    'ordinances for commercial wind energy conversion systems in {location}?"',
]
