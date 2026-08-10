"""COMPASS Geothermal Electricity plugin"""

import importlib.resources

from compass.plugin import create_schema_based_one_shot_extraction_plugin


COMPASSGeoElectricityExtractor = (
    create_schema_based_one_shot_extraction_plugin(
        importlib.resources.files("compass.extraction.geothermal_electricity")
        / "plugin_config.yaml",
        tech="geothermal",
    )
)
