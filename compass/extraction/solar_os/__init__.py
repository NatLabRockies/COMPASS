"""COMPASS Solar one-shot plugin"""

import importlib.resources

from compass.plugin import create_schema_based_one_shot_extraction_plugin


COMPASSSolarOneShotExtractor = create_schema_based_one_shot_extraction_plugin(
    importlib.resources.files("compass.extraction.solar_os")
    / "plugin_config.yaml",
    tech="solar-os",
)
