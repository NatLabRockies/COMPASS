"""COMPASS Geothermal Resource Management Plan plugin"""

import importlib.resources

from compass.plugin import create_schema_based_one_shot_extraction_plugin


COMPASSGeoRMPExtractor = create_schema_based_one_shot_extraction_plugin(
    importlib.resources.files("compass.extraction.rmp") / "plugin_config.yaml",
    tech="geo_rmp",
)


COMPASSGeoRMPExtractor.TEXT_COLLECTORS[0]._SP = """\
You are a structured extraction validator. You receive:
1) A text chunk.
2) An extraction schema that specifies the extraction criteria.

Determine whether the chunk contains content that matches any of the \
schema's criteria. Apply the schema's own inclusion rules faithfully: \
if the schema states that broad categories (e.g., oil/gas leasing, \
fluid mineral leasing, energy development) encompass the target \
technology, treat text addressing those broad categories as relevant. \
Do not require the target technology to be named verbatim when the \
schema explicitly defines broader qualifying criteria. If relevant, \
summarize the specific matching content; if not, state why it does \
not meet the schema's requirements. Keep the response concise and \
consistent.\
"""  # ruff:ignore[private-member-access]
