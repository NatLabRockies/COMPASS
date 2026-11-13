"""COMPASS data centers plugin"""

import importlib.resources

from compass.plugin import create_schema_based_one_shot_extraction_plugin


COMPASSDataCentersExtractor = create_schema_based_one_shot_extraction_plugin(
    importlib.resources.files("compass.extraction.data_centers")
    / "plugin_config.yaml",
    tech="data_centers",
)
