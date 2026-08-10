"""COMPASS wind extraction plugin"""

from compass.plugin import (
    OrdinanceExtractionPlugin,
    OutputColumn,
    register_plugin,
)
from compass.extraction.small_wind.ordinance import (
    SmallWindHeuristic,
    SmallWindOrdinanceTextCollector,
    SmallWindOrdinanceTextExtractor,
    SmallWindPermittedUseDistrictsTextCollector,
    SmallWindPermittedUseDistrictsTextExtractor,
)
from compass.extraction.small_wind.parse import (
    StructuredSmallWindOrdinanceParser,
    StructuredSmallWindPermittedUseDistrictsParser,
)

StructuredSmallWindOrdinanceParser.IN_LABEL = (
    SmallWindOrdinanceTextExtractor.OUT_LABEL
)
StructuredSmallWindPermittedUseDistrictsParser.IN_LABEL = (
    SmallWindPermittedUseDistrictsTextExtractor.OUT_LABEL
)

SMALL_WIND_QUERY_TEMPLATES = [
    "filetype:pdf {jurisdiction} wind energy conversion system ordinances",
    "wind energy conversion system ordinances {jurisdiction}",
    "{jurisdiction} wind WECS ordinance",
    (
        "Where can I find the legal text for small wind energy "
        "turbine zoning ordinances in {jurisdiction}?"
    ),
    (
        "What is the specific legal information regarding zoning "
        "ordinances for small wind turbines in {jurisdiction}?"
    ),
]

BEST_SMALL_WIND_ORDINANCE_WEBSITE_URL_KEYWORDS = {
    "pdf": 92160,
    "wecs": 46080,
    "wind": 23040,
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
}


class COMPASSSmallWindExtractor(OrdinanceExtractionPlugin):
    """COMPASS small wind extraction plugin"""

    IDENTIFIER = "small wind"
    """str: Identifier for extraction task """

    QUERY_TEMPLATES = SMALL_WIND_QUERY_TEMPLATES
    """list: List of search engine query templates for extraction"""

    WEBSITE_KEYWORDS = BEST_SMALL_WIND_ORDINANCE_WEBSITE_URL_KEYWORDS
    """list: List of keywords

    Keywords indicate links which should be prioritized when performing
    a website scrape for a wind ordinance document.
    """

    HEURISTIC = SmallWindHeuristic
    """BaseHeuristic: Class with a ``check()`` method"""

    TEXT_COLLECTORS = [
        SmallWindOrdinanceTextCollector,
        SmallWindPermittedUseDistrictsTextCollector,
    ]
    """Classes for collecting wind ordinance text chunks from docs"""

    TEXT_EXTRACTORS = [
        SmallWindOrdinanceTextExtractor,
        SmallWindPermittedUseDistrictsTextExtractor,
    ]
    """Class for extracting cleaned ord text from collected text"""

    PARSERS = [
        StructuredSmallWindOrdinanceParser,
        StructuredSmallWindPermittedUseDistrictsParser,
    ]
    """Class for parsing structured ordinance data from text"""

    OUTPUT_COLUMNS = [
        OutputColumn("county"),
        OutputColumn("state"),
        OutputColumn("subdivision"),
        OutputColumn("jurisdiction_type"),
        OutputColumn("FIPS"),
        OutputColumn("feature"),
        OutputColumn("value"),
        OutputColumn("units"),
        OutputColumn("adder"),
        OutputColumn("min_dist"),
        OutputColumn("max_dist"),
        OutputColumn("ordinance_text"),
        OutputColumn("explanation"),
        OutputColumn("year"),
        OutputColumn("section"),
        OutputColumn("source"),
        OutputColumn("quantitative"),
    ]
    """list: List of output columns for the extracted data"""


register_plugin(COMPASSSmallWindExtractor)
