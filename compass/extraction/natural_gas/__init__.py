"""COMPASS natural gas electric generating facilities plugin"""

import importlib.resources

from compass.plugin import create_schema_based_one_shot_extraction_plugin


COMPASSNaturalGasExtractor = create_schema_based_one_shot_extraction_plugin(
    importlib.resources.files("compass.extraction.natural_gas")
    / "plugin_config.yaml",
    tech="natural_gas",
)