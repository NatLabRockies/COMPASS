"""COMPASS plugin registry"""

from compass.utilities.jurisdictions import KNOWN_JURISDICTIONS_REGISTRY
from compass.plugin.base import BaseExtractionPlugin
from compass.exceptions import COMPASSPluginConfigurationError


PLUGIN_REGISTRY = {}
"""dict: Registered COMPASS plugins"""


def register_plugin(plugin_class):
    """Register a plugin class in the plugin registry

    Parameters
    ----------
    plugin_class : type
        The plugin class to register. Must be a subclass of
        :class:`~compass.plugin.base.BaseExtractionPlugin` and must pass
        the plugin configuration validation.

    Raises
    ------
    COMPASSPluginConfigurationError
        If the plugin class is not a subclass of
        :class:`~compass.plugin.base.BaseExtractionPlugin` or if it does
        not pass the plugin configuration validation.
    """
    if not issubclass(plugin_class, BaseExtractionPlugin):
        msg = (
            f"Plugin class {plugin_class.__name__} must be a subclass of "
            "`compass.plugin.base.BaseExtractionPlugin`!"
        )
        raise COMPASSPluginConfigurationError(msg)

    if plugin_class.JURISDICTION_DATA_FP is not None:
        KNOWN_JURISDICTIONS_REGISTRY.add(plugin_class.JURISDICTION_DATA_FP)

    plugin_instance = plugin_class(None, None)
    plugin_instance.validate_plugin_configuration()

    PLUGIN_REGISTRY[plugin_class.IDENTIFIER.casefold()] = plugin_class
