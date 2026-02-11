"""Schema-based cache for storing LLM-generated outputs"""

import json
import logging
import hashlib
from pathlib import Path

from platformdirs import user_data_dir


logger = logging.getLogger(__name__)
_CACHE_FP = "llm-generation_cache.json"
_SHA256_KEY = "sha256"


def key_from_cache(identifier, schema, key):
    """[NOT PUBLIC API] Get cached value for key/schema combination

    Parameters
    ----------
    identifier : str
        A string identifier for the technology of the extraction schema
        (e.g. "wind", "solar", "building_codes", etc.).
    schema : dict
        The extraction schema that is being used for the LLM-based
        one-shot extraction. This is used to ensure that cached content
        is only returned if the schema matches, which helps ensure
        that cached content is relevant and accurate for the current
        extraction task.
    key : str
        The specific key for the cached content to retrieve, (e.g.
        "query_templates", "website_keywords", etc.).

    Returns
    -------
    list or dict or None
        The cached value for the specified key/schema combination, or `
        `None`` if no valid cached value is found.
    """
    # cspell: disable-next-line
    data_dir = Path(user_data_dir(appname="INFRA-COMPASS", appauthor="NLR"))
    cache_fp = data_dir / _CACHE_FP
    cache = _load_cache(cache_fp)

    tech_cache = cache.get(identifier.casefold(), {})
    if not tech_cache:
        logger.debug("Did not find cache for %r", identifier)
        return None

    if tech_cache.get(_SHA256_KEY) != _schema_hash(schema):
        logger.debug(
            "Cache for %r exists but schema hash did not match", identifier
        )
        return None

    out = tech_cache.get(key)
    if not out:
        logger.debug(
            "Cache for %r exists and schema hash matches but no %r found",
            identifier,
            key,
        )
        return None

    logger.debug("Found %r for %r in cache:\n%r", key, identifier, out)
    return out


def key_to_cache(identifier, schema, key, value):
    """[NOT PUBLIC API] Cache key/value for given schema/tech combo

    Parameters
    ----------
    identifier : str
        A string identifier for the technology of the extraction schema
        (e.g. "wind", "solar", "building_codes", etc.).
    schema : dict
        The extraction schema that is being used for the LLM-based
        one-shot extraction. This is used to ensure that cached content
        is only returned if the schema matches, which helps ensure
        that cached content is relevant and accurate for the current
        extraction task.
    key : str
        The specific key for the cached content to retrieve, (e.g.
        "query_templates", "website_keywords", etc.).
    value : list or dict
        The value to cache for the specified key/schema combination.
        This should be the output of an LLM generation function that is
        being cached for future reuse. The value should be
        JSON-serializable since it will be stored in a JSON file on
        disk. Examples of values include a list of query templates for
        document retrieval, or a dictionary of website keywords and
        their relevance weights for link crawling prioritization. The
        value should be relevant to the technology and extraction task
        specified by the schema, and should be generated based on the
        content of the schema to ensure that it is useful and accurate
        for future extractions using the same schema.
    """
    # cspell: disable-next-line
    data_dir = Path(user_data_dir(appname="INFRA-COMPASS", appauthor="NLR"))
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_fp = data_dir / _CACHE_FP

    logger.debug("Loading query templates from cache at %s", cache_fp)
    cache = _load_cache(cache_fp)
    schema_hash = _schema_hash(schema)

    if identifier.casefold() not in cache:
        logger.debug(
            "Adding %r for %r to cache at %s",
            key,
            identifier,
            cache_fp,
        )
        cache[identifier.casefold()] = {key: value, _SHA256_KEY: schema_hash}
        _write_cache(cache_fp, cache)
        return

    potential_qt = cache[identifier.casefold()]
    if potential_qt.get(_SHA256_KEY) == schema_hash:
        logger.debug(
            "%r for %r already in cache and schema hash "
            "matches, so not updating cache",
            key,
            identifier,
        )
        return

    cache[identifier.casefold()] = {key: value, _SHA256_KEY: schema_hash}
    _write_cache(cache_fp, cache)


def _load_cache(cache_fp):
    """Load cache file contents as a dict"""
    if not cache_fp.exists():
        return {}

    logger.debug("Loading LLM generation cache at %s", cache_fp)
    return json.loads(cache_fp.read_text(encoding="utf-8"))


def _write_cache(cache_fp, cache):
    """Write cache file contents to disk"""
    cache_fp.write_text(json.dumps(cache, indent=4), encoding="utf-8")


def _schema_hash(schema):
    """Get SHA256 hash of the schema for cache validation"""
    m = hashlib.sha256()
    m.update(str(schema).encode())
    return m.hexdigest()
