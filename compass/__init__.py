"""Ordinance document download and structured data extraction"""

from dotenv import load_dotenv

from ._version import __version__
from .utilities.logs import setup_logging_levels, COMPASS_DEBUG_LEVEL

# Temporarily import to register plugins
# Can drop once plugins register themselves
from .extraction import (
    COMPASSGeoHeatPumpExtractor,
    COMPASSGeoElectricityExtractor,
    COMPASSNaturalGasExtractor,
    COMPASSNaturalGasPipelinesExtractor,
    COMPASSOilGasWellsExtractor,
    COMPASSGeoRMPExtractor,
    COMPASSSmallWindExtractor,
    COMPASSSolarExtractor,
    COMPASSTransmissionExtractor,
    COMPASSWindExtractor,
    COMPASSWindOneShotExtractor,
    TexasWaterRightsExtractor,
)

load_dotenv()
setup_logging_levels()
