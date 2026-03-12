"""Ordinance document download and structured data extraction"""

from ._version import __version__
from .utilities.logs import setup_logging_levels, COMPASS_DEBUG_LEVEL

# Temporarily import to register plugins
# Can drop once plugins register themselves
from .extraction import (
    COMPASSGeoHeatPumpExtractor,
    COMPASSSmallWindExtractor,
    COMPASSSolarExtractor,
    COMPASSWindExtractor,
    TexasWaterRightsExtractor,
)

setup_logging_levels()
