"""Ordinance document download and structured data extraction"""

import os

from ._version import __version__

COMPASS_DEBUG_LEVEL = int(os.environ.get("COMPASS_DEBUG_LEVEL", "0"))
