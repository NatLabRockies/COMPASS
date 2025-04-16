"""Custom Exceptions and Errors for COMPASS"""

import logging


logger = logging.getLogger("compass")


class COMPASSError(Exception):
    """Generic COMPASS Error"""

    def __init__(self, *args, **kwargs):
        """Init exception and broadcast message to logger"""
        super().__init__(*args, **kwargs)
        if args:
            logger.error(str(args[0]), stacklevel=2)


class COMPASSNotInitializedError(COMPASSError):
    """COMPASS not initialized error"""


class COMPASSValueError(COMPASSError, ValueError):
    """COMPASS ValueError"""


class COMPASSRuntimeError(COMPASSError, RuntimeError):
    """COMPASS RuntimeError"""
