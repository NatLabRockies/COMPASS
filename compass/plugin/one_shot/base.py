"""COMPASS one-shot extraction plugin"""

import json
import logging
import hashlib
import importlib.resources
from pathlib import Path

from platformdirs import user_data_dir

from compass.llm.calling import SchemaOutputLLMCaller
from compass.plugin import (
    register_plugin,
    NoOpHeuristic,
    NoOpTextCollector,
    NoOpTextExtractor,
    PromptBasedTextCollector,
    PromptBasedTextExtractor,
    OrdinanceExtractionPlugin,
)
from compass.plugin.one_shot.generators import generate_query_templates
from compass.plugin.one_shot.components import (
    SchemaBasedTextCollector,
    SchemaOrdinanceParser,
)
from compass.utilities.io import load_config
from compass.utilities.enums import LLMTasks


logger = logging.getLogger(__name__)
_SCHEMA_DIR = importlib.resources.files("compass.plugin.one_shot.schemas")


def create_schema_based_one_shot_extraction_plugin(config, tech):
    """Create a one-shot extraction plugin based on a configuration

    Parameters
    ----------
    config : dict or path-like
        One-shot configuration dictionary. If not a dictionary, should
        be a path to a file containing the configuration (supported
        formats: JSON, JSON5, YAML, TOML). See the wind ordinance schema
        for an example. The configuration must include the following
        keys:

            - `schema`: A dictionary representing the schema of the
               output. Can also be a path to a file that contains the
               schema (supported formats: JSON, JSON5, YAML, TOML). See
               the wind ordinance schema for an example.

        The configuration can also include the following optional keys:

            - `data_type_short_desc`: Short description of the type of
              data being extracted with this plugin, in the format
              `wind energy ordinance`, `solar energy ordinance`,
              `water rights`. This is used to enhance the prompts for
              the structured data extraction.
            - `query_templates`: A list of search engine query
              templates for document retrieval. Templates should include
              ``{jurisdiction}`` as a placeholder for the jurisdiction
              that is being processed. If not provided, the LLM will be
              used to generate search engine queries based on the
              schema input.
            - `website_keywords`: A list of keywords to use for
              filtering websites during document retrieval. If not
              provided, the LLM will be used to generate website
              keywords based on the schema input.
            - `collection_prompts`: A list of prompts to use for
              collecting relevant text from documents. Alternatively,
              this input can simply be ``True``, in which case the LLM
              will be used to generate the collection prompts. If
              ``False``, ``None``, or not provided, the entire document
              text will be used for extraction (no text collection).
            - `text_extraction_prompts`: A list of prompts to use for
              consolidating and extracting relevant text from the
              documents. Alternatively, this input can simply be
              ``True``, in which case the LLM will be used to generate
              the text extraction prompts. If ``False``, ``None``, or
              not provided, the entire document text will be used for
              extraction (no text consolidation).
            - `cache_query_templates`: Boolean flag indicating
              whether or not to cache generated query templates and
              website keywords for future use. By default, ``True``.
              Caching is recommended since the generation of query
              templates and website keywords can be costly, but if you
              are iterating on the configuration and want to see the
              effect of changes to the schema on the generated query
              templates and website keywords in real time, you may want
              to set this flag to ``False`` to avoid caching generated
              templates/keywords until you have finalized the schema.
            - `extraction_system_prompt`: Custom system prompt to use
              for the structured data extraction step. If not provided,
              a default prompt will be used that instructs the LLM to
              extract structured data from the given document(s). You
              may provide a custom system prompt if you want to provide
              more specific instructions to the LLM for the structured
              data extraction step.

    tech : str
        Technology identifier to use for the plugin (e.g., "wind",
        "solar"). Must be unique from the identifiers of any existing
        plugins.
    """
    if not isinstance(config, dict):
        config = load_config(config)

    if isinstance(config["schema"], str):
        config["schema"] = load_config(config["schema"])

    text_collectors = _collectors_from_config(config)
    text_extractors = _extractors_from_config(
        config, in_label=text_collectors[-1].OUT_LABEL
    )
    parsers = _parser_from_config(
        config, in_label=text_extractors[-1].OUT_LABEL
    )

    class SchemaBasedExtractionPlugin(OrdinanceExtractionPlugin):
        SCHEMA = config["schema"]
        """dict: Schema for the output of the text extraction step"""

        IDENTIFIER = tech
        """str: Identifier for extraction task """

        # TODO: implement dynamic generation of the heuristic based on
        # the extraction schema
        HEURISTIC = NoOpHeuristic
        """BaseHeuristic: Class with a ``check()`` method"""

        # TODO: implement dynamic generation of the website keywords
        # based on the extraction schema
        WEBSITE_KEYWORDS = {
            "pdf": 23040,
            "zoning": 11520,
            "ordinance": 5760,
            r"renewable%20energy": 1440,
            r"renewable+energy": 1440,
            "renewable energy": 1440,
            "planning": 720,
            "plan": 360,
            "government": 180,
            "code": 60,
            "area": 60,
            r"land%20development": 15,
            r"land+development": 15,
            "land development": 15,
            "land": 3,
            "environment": 3,
            "energy": 3,
            "renewable": 3,
            "municipal": 1,
            "department": 1,
        }

        TEXT_COLLECTORS = text_collectors
        """Classes for collecting text chunks from docs"""

        TEXT_EXTRACTORS = text_extractors
        """Classes for extracting cleaned text from collected text"""

        PARSERS = parsers
        """Classes for parsing structured ordinance data from text"""

        QUERY_TEMPLATES = []  # set by user or LLM-generated
        """List: List of search engine query templates"""

        async def get_query_templates(self):
            """Get a list of query templates for document retrieval

            Returns
            -------
            list
                List of search engine query templates for document
                retrieval. Templates may include ``{jurisdiction}`` as
                a placeholder for the jurisdiction that is being
                processed.
            """
            if self.QUERY_TEMPLATES:
                return self.QUERY_TEMPLATES

            if qt := config.get("query_templates"):
                self.QUERY_TEMPLATES = qt
                return qt

            qt = _qt_from_cache(self.IDENTIFIER, config["schema"])
            if qt:
                self.QUERY_TEMPLATES = qt
                return qt

            model_config = self.model_configs.get(
                LLMTasks.PLUGIN_GENERATION,
                self.model_configs[LLMTasks.DEFAULT],
            )
            schema_llm = SchemaOutputLLMCaller(
                llm_service=model_config.llm_service,
                usage_tracker=self.usage_tracker,
                **model_config.llm_call_kwargs,
            )
            logger.debug("Generating query templates...")
            qt = await generate_query_templates(
                schema_llm, config["schema"], add_think_prompt=True
            )
            logger.debug("Generated the following query templates:\n%r", qt)
            self.QUERY_TEMPLATES = qt

            if config.get("cache_query_templates", True):
                _qt_to_cache(self.IDENTIFIER, config["schema"], qt)

            return qt

        def _validate_query_templates(self):
            """NoOp validation for query templates

            Since templates can be generated by LLM, we don't know until
            runtime whether or not they will be valid.
            """

    register_plugin(SchemaBasedExtractionPlugin)


def _collectors_from_config(config):
    """Create a TextCollector subclass based on a config dict"""
    cp = config.get("collection_prompts")

    if cp is True:
        schema_fp = _SCHEMA_DIR / "validate_chunk.json5"

        class PluginCollector(SchemaBasedTextCollector):
            OUT_LABEL = NoOpTextCollector.OUT_LABEL  # reuse label
            SCHEMA = config["schema"]
            OUTPUT_SCHEMA = load_config(schema_fp)

        return [PluginCollector]

    if cp:

        class PluginCollector(PromptBasedTextCollector):
            OUT_LABEL = NoOpTextCollector.OUT_LABEL  # reuse label
            PROMPTS = cp

        return [PluginCollector]

    return [NoOpTextCollector]


def _extractors_from_config(config, in_label):
    """Create a TextExtractor subclass based on a config dict"""
    tep = config.get("text_extraction_prompts")

    if tep is True:
        # TODO: When implementing this, don't forget to register the
        # text output file name so it gets store in the
        # cleaned outputs directory
        msg = (
            "LLM-based text extraction not implemented yet. If you would like "
            "to see this feature implemented, please submit an issue or, "
            "better yet, a pull request!"
        )
        raise NotImplementedError(msg)

    if tep:

        class PluginTextExtractor(PromptBasedTextExtractor):
            IN_LABEL = in_label
            PROMPTS = tep

        return [PluginTextExtractor]

    class PluginTextExtractor(NoOpTextExtractor):
        IN_LABEL = in_label
        OUT_LABEL = "copied_relevant_text"

    return [PluginTextExtractor]


def _parser_from_config(config, in_label):
    """Create a TextExtractor subclass based on a config dict"""

    new_sys_prompt = config.get(
        "extraction_system_prompt", SchemaOrdinanceParser.SYSTEM_PROMPT
    )

    class PluginParser(SchemaOrdinanceParser):
        IN_LABEL = in_label
        OUT_LABEL = "structured_data"
        SCHEMA = config["schema"]
        DATA_TYPE_SHORT_DESC = config.get("data_type_short_desc")
        SYSTEM_PROMPT = new_sys_prompt

    return [PluginParser]


def _qt_from_cache(identifier, schema):
    """Get cached query templates for a given schema if they exist"""
    # cspell: disable-next-line
    data_dir = Path(user_data_dir(appname="INFRA-COMPASS", appauthor="NLR"))
    cache_fp = data_dir / "qt_cache.json"
    if not cache_fp.exists():
        return None

    logger.debug("Loading query templates from cache at %s", cache_fp)
    qt = json.loads(cache_fp.read_text(encoding="utf-8"))
    if identifier.casefold() not in qt:
        return None

    potential_qt = qt[identifier.casefold()]
    m = hashlib.sha256()
    m.update(str(schema).encode())
    if potential_qt.get("sha256") != m.hexdigest():
        return None

    templates = potential_qt["templates"]
    logger.debug(
        "Found query templates for %r in cache:\n%r", identifier, templates
    )
    return templates


def _qt_to_cache(identifier, schema, qt):
    """Cache generated query templates for future use"""
    # cspell: disable-next-line
    data_dir = Path(user_data_dir(appname="INFRA-COMPASS", appauthor="NLR"))
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_fp = data_dir / "qt_cache.json"
    if not cache_fp.exists():
        logger.debug(
            "Cache file for query templates not found at %s. Creating new "
            "cache with current query templates for %r",
            cache_fp,
            identifier,
        )
        cache = {
            identifier.casefold(): {
                "templates": qt,
                "sha256": hashlib.sha256(str(schema).encode()).hexdigest(),
            }
        }
        cache_fp.write_text(json.dumps(cache, indent=4), encoding="utf-8")
        return

    logger.debug("Loading query templates from cache at %s", cache_fp)
    cache = json.loads(cache_fp.read_text(encoding="utf-8"))
    if identifier.casefold() not in qt:
        logger.debug(
            "Adding query templates for %r to cache at %s",
            identifier,
            cache_fp,
        )
        cache[identifier.casefold()] = {
            "templates": qt,
            "sha256": hashlib.sha256(str(schema).encode()).hexdigest(),
        }
        cache_fp.write_text(json.dumps(cache, indent=4), encoding="utf-8")
        return

    potential_qt = qt[identifier.casefold()]
    m = hashlib.sha256()
    m.update(str(schema).encode())
    if potential_qt.get("sha256") == m.hexdigest():
        logger.debug(
            "Query templates for %r already in cache and schema hash "
            "matches, so not updating cache",
            identifier,
        )
        return

    cache[identifier.casefold()] = {
        "templates": qt,
        "sha256": hashlib.sha256(str(schema).encode()).hexdigest(),
    }
    cache_fp.write_text(json.dumps(cache, indent=4), encoding="utf-8")
    return
