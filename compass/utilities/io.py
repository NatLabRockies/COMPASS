"""COMPASS I/O utilities

A lot of this is taken directly from NLR's GAPs repo:
https://github.com/NatLabRockies/gaps
"""

import logging
import contextlib
import collections
from copy import deepcopy
from pathlib import Path
from abc import ABC, abstractmethod

import json
import yaml
import toml
import pyjson5

from compass.utilities.enums import CaseInsensitiveEnum
from compass.exceptions import COMPASSValueError, COMPASSFileNotFoundError


logger = logging.getLogger(__name__)
_CONFIG_HANDLER_REGISTRY = {}
_INHERIT_FROM_KEY = "inherit_from"
_DELETE_SENTINEL = "DELETE"


class _JSON5Formatter:
    """Format input JSON5 data with indentation"""

    def __init__(self, data):
        self.data = data

    def _format_as_json(self):
        """Format the data input with as string with indentation"""
        return json.dumps(self.data, indent=4)


class Handler(ABC):
    """ABC for configuration file handler"""

    def __init_subclass__(cls):
        super().__init_subclass__()
        if isinstance(cls.FILE_EXTENSION, str):
            _CONFIG_HANDLER_REGISTRY[cls.FILE_EXTENSION] = cls
        else:
            for file_extension in cls.FILE_EXTENSION:
                _CONFIG_HANDLER_REGISTRY[file_extension] = cls

    @classmethod
    def load(cls, file_name):
        """Load the file contents"""
        config_str = Path(file_name).read_text(encoding="utf-8")
        return cls.loads(config_str)

    @classmethod
    def write(cls, file_name, data):
        """Write the data to a file"""
        with Path(file_name).open("w", encoding="utf-8") as config_file:
            cls.dump(data, config_file)

    @classmethod
    @abstractmethod
    def dump(cls, config, stream):
        """Write the config to a stream (file)"""

    @classmethod
    @abstractmethod
    def dumps(cls, config):
        """Convert the config to a string"""

    @classmethod
    @abstractmethod
    def loads(cls, config_str):
        """Parse the string into a config dictionary"""

    @property
    @abstractmethod
    def FILE_EXTENSION(self):  # ruff:ignore[invalid-function-name]
        """str: Enum name to use"""


class JSONHandler(Handler):
    """JSON config file handler"""

    FILE_EXTENSION = "json"
    """JSON file extension"""

    @classmethod
    def dump(cls, config, stream):
        """Write the config to a stream (JSON file)"""
        return json.dump(config, stream, indent=4)

    @classmethod
    def dumps(cls, config):
        """Convert the config to a JSON string"""
        return json.dumps(config, indent=4)

    @classmethod
    def loads(cls, config_str):
        """Parse the JSON string into a config dictionary"""
        return json.loads(config_str)


class JSON5Handler(Handler):
    """JSON5 config file handler"""

    FILE_EXTENSION = "json5"
    """JSON5 file extension"""

    @classmethod
    def dump(cls, config, stream):
        """Write the config to a stream (JSON5 file)"""
        return pyjson5.encode_io(
            _JSON5Formatter(config),
            stream,
            supply_bytes=False,
            tojson="_format_as_json",
        )

    @classmethod
    def dumps(cls, config):
        """Convert the config to a JSON5 string"""
        return pyjson5.encode(
            _JSON5Formatter(config),
            tojson="_format_as_json",
        )

    @classmethod
    def loads(cls, config_str):
        """Parse the JSON5 string into a config dictionary"""
        return pyjson5.decode(config_str, maxdepth=-1)


class YAMLHandler(Handler):
    """YAML config file handler"""

    FILE_EXTENSION = "yaml", "yml"
    """YAML file extensions"""

    @classmethod
    def dump(cls, config, stream):
        """Write the config to a stream (YAML file)"""
        return yaml.safe_dump(config, stream, indent=2, sort_keys=False)

    @classmethod
    def dumps(cls, config):
        """Convert the config to a YAML string"""
        return yaml.safe_dump(config, indent=2, sort_keys=False)

    @classmethod
    def loads(cls, config_str):
        """Parse the YAML string into a config dictionary"""
        return yaml.safe_load(config_str)


class TOMLHandler(Handler):
    """TOML config file handler"""

    FILE_EXTENSION = "toml"
    """TOML file extension"""

    @classmethod
    def dump(cls, config, stream):
        """Write the config to a stream (TOML file)"""
        return toml.dump(config, stream)

    @classmethod
    def dumps(cls, config):
        """Convert the config to a TOML string"""
        return toml.dumps(config)

    @classmethod
    def loads(cls, config_str):
        """Parse the TOML string into a config dictionary"""
        return toml.loads(config_str)


class _ConfigType(CaseInsensitiveEnum):
    """Base config type enum class only meant to be initialized once"""

    @classmethod
    def _new_post_hook(cls, obj, value):
        """Hook for post-processing after __new__; adds methods"""
        obj.dump = _CONFIG_HANDLER_REGISTRY[value].dump
        obj.dumps = _CONFIG_HANDLER_REGISTRY[value].dumps
        obj.load = _CONFIG_HANDLER_REGISTRY[value].load
        obj.loads = _CONFIG_HANDLER_REGISTRY[value].loads
        obj.write = _CONFIG_HANDLER_REGISTRY[value].write
        obj.__doc__ = f"{value.upper()} config file handler"
        return obj


ConfigType = _ConfigType(
    "ConfigType",
    {
        config_type.upper(): config_type
        for config_type in _CONFIG_HANDLER_REGISTRY
    },
)
"""An enumeration of the parseable config types"""


def load_config(
    config_filepath,
    resolve_paths=True,
    file_name="Configuration",
    excluded_keys=None,
):
    """Load a config file, recursively applying inherited configuration

    Parameters
    ----------
    config_filepath : path-like
        Path to config file.
    resolve_paths : bool, default=True
        Option to (recursively) resolve file-paths in the dictionary
        w.r.t the config file directory.
        By default, ``True``.
    file_name : str, default="Configuration"
        Name of the config file for error messages.
        By default, "Configuration".
    excluded_keys : collection of str, optional
        Dictionary keys whose values should not be resolved as paths.
        By default, ``None``.

    Returns
    -------
    dict
        Dictionary containing configuration parameters.

    Raises
    ------
    COMPASSValueError
        If the file extension is missing or unsupported, or inheritance
        is invalid or circular.
    COMPASSFileNotFoundError
        If a config file does not exist.

    Notes
    -----
    A config may use ``inherit_from`` to name another config file.
    Parent configs are loaded recursively and dictionaries are
    deep-merged, with child values taking precedence. Lists are
    replaced, not combined. An exact child value of ``"DELETE"`` removes
    that key from the merged result. Parent references are relative to
    the file where they are defined. When path resolution is enabled,
    other relative paths are also resolved from their defining file.
    """
    return _load_config(
        config_filepath,
        resolve_paths=resolve_paths,
        file_name=file_name,
        excluded_keys=excluded_keys,
        inheritance_chain=(),
    )


def _load_config(
    config_filepath, resolve_paths, file_name, excluded_keys, inheritance_chain
):
    """Recursively load and merge a config inheritance chain"""
    config_filepath = Path(config_filepath).expanduser().resolve()
    _ensure_no_circular_inheritance(config_filepath, inheritance_chain)

    config = _read_config_file(config_filepath, file_name)
    has_inheritance = (
        isinstance(config, collections.abc.Mapping)
        and _INHERIT_FROM_KEY in config
    )
    if has_inheritance:
        inherit_from = config.pop(_INHERIT_FROM_KEY)
        _validate_inheritance_input(inherit_from, file_name, config_filepath)

    if resolve_paths:
        config = resolve_all_paths(
            config, config_filepath.parent, excluded_keys=excluded_keys
        )

    if not has_inheritance:
        return config

    parent_filepath = Path(inherit_from.replace("\\", "/")).expanduser()
    if not parent_filepath.is_absolute():
        parent_filepath = config_filepath.parent / parent_filepath

    parent_config = _load_config(
        parent_filepath,
        resolve_paths=resolve_paths,
        file_name=file_name,
        excluded_keys=excluded_keys,
        inheritance_chain=(*inheritance_chain, config_filepath),
    )
    _validate_inherited_config(parent_config, file_name, parent_filepath)

    return _merge_configs(parent_config, config)


def _read_config_file(config_filepath, file_name):
    """Validate a config file path and load its contents"""
    if "." not in config_filepath.name:
        msg = (
            f"{file_name} file must have a file-ending. Got: "
            f"{config_filepath.name}"
        )
        raise COMPASSValueError(msg)

    if not config_filepath.exists():
        msg = f"{file_name} file does not exist: {config_filepath}"
        raise COMPASSFileNotFoundError(msg)

    try:
        config_type = ConfigType(config_filepath.suffix[1:])
    except ValueError as err:
        msg = (
            f"Got unknown config file extension: "
            f"{config_filepath.suffix!r}. Supported extensions are: "
            f"{', '.join({ct.value for ct in ConfigType})}"
        )
        raise COMPASSValueError(msg) from err

    return config_type.load(config_filepath)


def _ensure_no_circular_inheritance(config_filepath, inheritance_chain):
    """Ensure there is no circular inheritance in the config files"""
    if config_filepath in inheritance_chain:
        cycle = (*inheritance_chain, config_filepath)
        chain = " -> ".join(path.as_posix() for path in cycle)
        msg = f"Circular config inheritance detected: {chain}"
        raise COMPASSValueError(msg)


def _validate_inheritance_input(inherit_from, file_name, config_filepath):
    """Ensure the inheritance input is valid"""
    if not isinstance(inherit_from, str) or not inherit_from.strip():
        msg = (
            f"{file_name} inheritance key {_INHERIT_FROM_KEY!r} in "
            f"{config_filepath.as_posix()!r} must be a non-empty string"
        )
        raise COMPASSValueError(msg)


def _validate_inherited_config(parent_config, file_name, parent_filepath):
    """Ensure the inherited config is a mapping"""
    if not isinstance(parent_config, collections.abc.Mapping):
        msg = (
            f"{file_name} inherited config must be a mapping: "
            f"{parent_filepath}"
        )
        raise COMPASSValueError(msg)


def _merge_configs(parent, child):
    """Deep merge a child config over a parent config"""
    merged = deepcopy(parent)
    for key, value in child.items():
        if value == _DELETE_SENTINEL:
            merged.pop(key, None)
        elif isinstance(value, collections.abc.Mapping):
            parent_value = merged.get(key, {})
            if not isinstance(parent_value, collections.abc.Mapping):
                parent_value = {}
            merged[key] = _merge_configs(parent_value, value)
        else:
            merged[key] = deepcopy(value)
    return merged


# complexipy: ignore
def resolve_all_paths(container, base_dir, excluded_keys=None):
    """Perform a deep string replacement and path resolve in `container`

    Parameters
    ----------
    container : dict or list
        Container like a dictionary or list that may (or may not)
        contain relative paths to resolve.
    base_dir : path-like
        Base path to directory from which to resolve path string
        (typically current directory)
    excluded_keys : collection of str, optional
        Dictionary keys whose values should not be resolved as paths.
        By default, ``None``.

    Returns
    -------
    dict or list
        Input container with updated strings.
    """

    excluded_keys = set(excluded_keys or ())

    if isinstance(container, str):
        # `resolve_path` is safe to call on any string,
        # even if it is not a path
        container = resolve_path(container, Path(base_dir))

    elif isinstance(container, collections.abc.Mapping):
        container = {
            key: (
                val
                if key in excluded_keys
                else resolve_all_paths(
                    val, Path(base_dir), excluded_keys=excluded_keys
                )
            )
            for key, val in container.items()
        }

    elif isinstance(container, collections.abc.Sequence):
        container = [
            resolve_all_paths(
                item, Path(base_dir), excluded_keys=excluded_keys
            )
            for item in container
        ]

    return container


def resolve_path(path, base_dir):
    """Resolve a file path represented by the input string.

    This function resolves the input string if it resembles a path.
    Specifically, the string will be resolved if it starts  with
    "``./``" or "``..``", or it if it contains either "``./``" or
    "``..``" somewhere in the string body. Otherwise, the string
    is returned unchanged, so this function *is* safe to call on any
    string, even ones that do not resemble a path.
    This method delegates the "resolving" logic to
    :meth:`pathlib.Path.resolve`. This means the path is made
    absolute, symlinks are resolved, and "``..``" components are
    eliminated. If the ``path`` input starts with "``./``" or
    "``..``", it is assumed to be w.r.t the config directory, *not*
    the run directory.

    Parameters
    ----------
    path : str
        Input file path.
    base_dir : path-like
        Base path to directory from which to resolve path string
        (typically current directory).

    Returns
    -------
    str
        The resolved path.
    """
    base_dir = Path(base_dir)
    normalized = path.replace("\\", "/")

    if normalized.startswith("./"):
        path = base_dir / Path(normalized[2:])
    elif normalized.startswith(".."):
        path = base_dir / Path(normalized)
    elif (
        "/./" in normalized
        or normalized.endswith("/.")
        or ("/../" in normalized or normalized.endswith("/.."))
    ):
        path = Path(normalized)

    with contextlib.suppress(AttributeError):
        path = path.expanduser().resolve().as_posix()

    return path


def normalize_output_stem(out_stem):
    """[NOT PUBLIC API] Normalize an output file name stem"""
    return (
        out_stem.replace(".", "")
        .replace(",", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )
