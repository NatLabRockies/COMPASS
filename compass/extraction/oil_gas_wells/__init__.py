"""COMPASS Oil and Gas Wells plugin"""

import importlib.resources

from compass.plugin import create_schema_based_one_shot_extraction_plugin


COMPASSOilGasWellsExtractor = create_schema_based_one_shot_extraction_plugin(
    importlib.resources.files("compass.extraction.oil_gas_wells")
    / "plugin_config.yaml",
    tech="oil_gas_wells",
)
