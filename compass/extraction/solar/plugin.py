"""COMPASS solar extraction plugin"""

from compass.plugin import (
    OrdinanceExtractionPlugin,
    OutputColumn,
    register_plugin,
    normalize_website_keywords,
)
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


BEST_SOLAR_ORDINANCE_WEBSITE_URL_KEYWORDS = [
    "pdf",
    "secs",
    "solar",
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


class COMPASSSolarExtractor(OrdinanceExtractionPlugin):
    """COMPASS solar extraction plugin"""

    IDENTIFIER = "solar"
    """str: Identifier for extraction task """

    QUERY_TEMPLATES = SOLAR_QUERY_TEMPLATES
    """list: List of search engine query templates for extraction"""

    WEBSITE_KEYWORDS = normalize_website_keywords(
        BEST_SOLAR_ORDINANCE_WEBSITE_URL_KEYWORDS
    )
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


register_plugin(COMPASSSolarExtractor)
