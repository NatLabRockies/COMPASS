"""COMPASS Geothermal ground source heat pump plugin"""

import importlib.resources

from compass.plugin import create_schema_based_one_shot_extraction_plugin


COMPASSGeoHeatPumpExtractor = create_schema_based_one_shot_extraction_plugin(
    importlib.resources.files("compass.extraction.ghp") / "plugin_config.yaml",
    tech="ghp",
)
