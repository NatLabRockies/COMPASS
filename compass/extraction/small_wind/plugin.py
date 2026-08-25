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


register_plugin(COMPASSSmallWindExtractor)
