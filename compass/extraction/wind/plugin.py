"""COMPASS wind extraction plugin"""

from compass.plugin import (
    OrdinanceExtractionPlugin,
    OutputColumn,
    register_plugin,
    normalize_website_keywords,
)
from compass.extraction.wind.ordinance import (
    WindHeuristic,
    WindOrdinanceTextCollector,
    WindOrdinanceTextExtractor,
    WindPermittedUseDistrictsTextCollector,
    WindPermittedUseDistrictsTextExtractor,
)
from compass.extraction.wind.parse import (
    StructuredWindOrdinanceParser,
    StructuredWindPermittedUseDistrictsParser,
)

StructuredWindOrdinanceParser.IN_LABEL = WindOrdinanceTextExtractor.OUT_LABEL
StructuredWindPermittedUseDistrictsParser.IN_LABEL = (
    WindPermittedUseDistrictsTextExtractor.OUT_LABEL
)

WIND_QUERY_TEMPLATES = [
    "filetype:pdf {jurisdiction} wind energy conversion system ordinances",
    "wind energy conversion system ordinances {jurisdiction}",
    "{jurisdiction} wind WECS ordinance",
    (
        "Where can I find the legal text for commercial wind energy "
        "conversion system zoning ordinances in {jurisdiction}?"
    ),
    (
        "What is the specific legal information regarding zoning "
        "ordinances for commercial wind energy conversion systems in "
        "{jurisdiction}?"
    ),
]

BEST_WIND_ORDINANCE_WEBSITE_URL_KEYWORDS = [
    "pdf",
    "wecs",
    "wind",
    ["zoning", "ordinance", "regulation"],
    ["dsireusa", "windaction"],
    ["codelibrary", "amlegal", "municode", "codepublishing", "ecode360"],
    "renewable energy",
    ["plan", "planning", "permit"],
    "government",
    ["setback", "noise"],
    ["code", "area"],
    ["land development", "land use"],
    ["land", "environment", "energy", "renewable"],
    ["municipal", "department", "development", "board"],
]


class COMPASSWindExtractor(OrdinanceExtractionPlugin):
    """COMPASS wind extraction plugin"""

    IDENTIFIER = "wind"
    """str: Identifier for extraction task """

    QUERY_TEMPLATES = WIND_QUERY_TEMPLATES
    """list: List of search engine query templates for extraction"""

    WEBSITE_KEYWORDS = normalize_website_keywords(
        BEST_WIND_ORDINANCE_WEBSITE_URL_KEYWORDS
    )
    """list: List of keywords

    Keywords indicate links which should be prioritized when performing
    a website scrape for a wind ordinance document.
    """

    HEURISTIC = WindHeuristic
    """BaseHeuristic: Class with a ``check()`` method"""

    TEXT_COLLECTORS = [
        WindOrdinanceTextCollector,
        WindPermittedUseDistrictsTextCollector,
    ]
    """Classes for collecting wind ordinance text chunks from docs"""

    TEXT_EXTRACTORS = [
        WindOrdinanceTextExtractor,
        WindPermittedUseDistrictsTextExtractor,
    ]
    """Class for extracting cleaned ord text from collected text"""

    PARSERS = [
        StructuredWindOrdinanceParser,
        StructuredWindPermittedUseDistrictsParser,
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
        OutputColumn("units", include_in_qual_output=False),
        OutputColumn("adder", include_in_qual_output=False),
        OutputColumn("min_dist", include_in_qual_output=False),
        OutputColumn("max_dist", include_in_qual_output=False),
        OutputColumn("summary"),
        OutputColumn("year"),
        OutputColumn("section"),
        OutputColumn("source"),
        OutputColumn(
            "quantitative",
            include_in_quant_output=False,
            include_in_qual_output=False,
        ),
    ]
    """list: List of output columns for the extracted data"""


register_plugin(COMPASSWindExtractor)
