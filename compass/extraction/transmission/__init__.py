"""COMPASS high-voltage transmission extraction plugin"""

import importlib.resources

from compass.plugin import create_schema_based_one_shot_extraction_plugin


COMPASSTransmissionExtractor = create_schema_based_one_shot_extraction_plugin(
    importlib.resources.files("compass.extraction.transmission")
    / "plugin_config.yaml",
    tech="transmission",
)
