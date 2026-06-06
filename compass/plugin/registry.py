"""COMPASS plugin registry"""

from compass.utilities.jurisdictions import KNOWN_JURISDICTIONS_REGISTRY
from compass.plugin.base import BaseExtractionPlugin
from compass.exceptions import (
    COMPASSPluginConfigurationError,
    COMPASSValueError,
)


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

    if (plugin_id := plugin_class.IDENTIFIER.casefold()) in PLUGIN_REGISTRY:
        msg = (
            f"Plugin identifier '{plugin_class.IDENTIFIER}' is already in "
            "use by another plugin! Please choose a unique identifier for "
            f"{plugin_class.__name__}."
        )
        raise COMPASSPluginConfigurationError(msg)

    plugin_class(None, None).validate_plugin_configuration()

    if plugin_class.JURISDICTION_DATA_FP is not None:
        KNOWN_JURISDICTIONS_REGISTRY.add(plugin_class.JURISDICTION_DATA_FP)
    PLUGIN_REGISTRY[plugin_id] = plugin_class


def resolve_plugin(tech):
    """Look up the registered plugin class for a technology

    Parameters
    ----------
    tech : str
        Technology name to look up. The lookup is case-insensitive and
        based on the plugin class's ``IDENTIFIER`` attribute.

    Returns
    -------
    type
        The plugin class registered for the given technology.

    Raises
    ------
    COMPASSValueError
        If no plugin is registered for the given technology.
    """
    if (plugin_cls := PLUGIN_REGISTRY.get(tech.casefold())) is not None:
        return plugin_cls

    msg = (
        f"No plugin registered for tech={tech!r}. Available: "
        f"{sorted(PLUGIN_REGISTRY)}"
    )
    raise COMPASSValueError(msg)
