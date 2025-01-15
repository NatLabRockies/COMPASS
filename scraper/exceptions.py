"""Custom Exceptions and Errors for Ordinances"""

import logging


logger = logging.getLogger("scraper")


class OrdinanceError(Exception):
    """Generic Ordinance Error"""

    def __init__(self, *args, **kwargs):
        """Init exception and broadcast message to logger"""
        super().__init__(*args, **kwargs)
        if args:
            logger.error(str(args[0]), stacklevel=2)


class OrdinanceNotInitializedError(OrdinanceError):
    """Ordinances not initialized error"""


class OrdinanceValueError(OrdinanceError, ValueError):
    """Ordinances ValueError"""
