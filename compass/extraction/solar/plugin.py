"""COMPASS solar extraction plugin"""

from compass.plugin import OrdinanceExtractionPlugin, register_plugin
from compass.extraction.solar.ordinance import (
    SolarHeuristic,
    SolarOrdinanceTextCollector,
    SolarOrdinanceTextExtractor,
    SolarPermittedUseDistrictsTextCollector,
    SolarPermittedUseDistrictsTextExtractor,
)
from compass.extraction.solar.parse import (
    StructuredSolarOrdinanceParser,
    StructuredSolarPermittedUseDistrictsParser,
)

StructuredSolarOrdinanceParser.IN_LABEL = SolarOrdinanceTextExtractor.OUT_LABEL
StructuredSolarPermittedUseDistrictsParser.IN_LABEL = (
    SolarPermittedUseDistrictsTextExtractor.OUT_LABEL
)

SOLAR_QUERY_TEMPLATES = [
    "filetype:pdf {jurisdiction} solar energy conversion system ordinances",
    "solar energy conversion system ordinances {jurisdiction}",
    "{jurisdiction} solar energy farm ordinance",
    (
        "Where can I find the legal text for commercial solar energy "
        "conversion system zoning ordinances in {jurisdiction}?"
    ),
    (
        "What is the specific legal information regarding zoning "
        "ordinances for commercial solar energy conversion systems in "
        "{jurisdiction}?"
    ),
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


class COMPASSSolarExtractor(OrdinanceExtractionPlugin):
    """COMPASS solar extraction plugin"""

    IDENTIFIER = "solar"
    """str: Identifier for extraction task """

    QUERY_TEMPLATES = SOLAR_QUERY_TEMPLATES
    """list: List of search engine query templates for extraction"""

    WEBSITE_KEYWORDS = BEST_SOLAR_ORDINANCE_WEBSITE_URL_KEYWORDS
    """list: List of keywords

    Keywords indicate links which should be prioritized when performing
    a website scrape for a wind ordinance document.
    """

    HEURISTIC = SolarHeuristic
    """BaseHeuristic: Class with a ``check()`` method"""

    TEXT_COLLECTORS = [
        SolarOrdinanceTextCollector,
        SolarPermittedUseDistrictsTextCollector,
    ]
    """Classes for collecting wind ordinance text chunks from docs"""

    TEXT_EXTRACTORS = [
        SolarOrdinanceTextExtractor,
        SolarPermittedUseDistrictsTextExtractor,
    ]
    """Class for extracting cleaned ord text from collected text"""

    PARSERS = [
        StructuredSolarOrdinanceParser,
        StructuredSolarPermittedUseDistrictsParser,
    ]
    """Class for parsing structured ordinance data from text"""


register_plugin(COMPASSSolarExtractor)
