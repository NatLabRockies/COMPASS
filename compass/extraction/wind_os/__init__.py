"""COMPASS Wind one-shot plugin"""

import importlib.resources

from compass.plugin import create_schema_based_one_shot_extraction_plugin


COMPASSWindOneShotExtractor = create_schema_based_one_shot_extraction_plugin(
    importlib.resources.files("compass.extraction.wind_os")
    / "plugin_config.yaml",
    tech="wind-os",
)
