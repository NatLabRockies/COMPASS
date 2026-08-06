"""COMPASS natural gas pipelines and compressor stations plugin"""

import importlib.resources

from compass.plugin import create_schema_based_one_shot_extraction_plugin


COMPASSNaturalGasPipelinesExtractor = (
    create_schema_based_one_shot_extraction_plugin(
        importlib.resources.files("compass.extraction.natural_gas_pipelines")
        / "plugin_config.yaml",
        tech="natural_gas_pipelines",
    )
)
