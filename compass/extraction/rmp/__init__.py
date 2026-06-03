"""COMPASS Geothermal Resource Management Plan plugin"""

import importlib.resources

from compass.plugin import create_schema_based_one_shot_extraction_plugin


COMPASSGeoRMPExtractor = create_schema_based_one_shot_extraction_plugin(
    importlib.resources.files("compass.extraction.rmp") / "plugin_config.yaml",
    tech="geo_rmp",
)
