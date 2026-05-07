"""COMPASS one-shot extraction plugin"""

import logging
import importlib.resources
from asyncio import Semaphore
from enum import StrEnum, auto

from compass.llm.calling import SchemaOutputLLMCaller
from compass.plugin import (
    register_plugin,
    NoOpHeuristic,
    NoOpTextCollector,
    NoOpTextExtractor,
    DocSelectionMethod,
    PromptBasedTextCollector,
    PromptBasedTextExtractor,
    OrdinanceExtractionPlugin,
    KeywordBasedHeuristic,
)
from compass.plugin.one_shot.generators import (
    generate_query_templates,
    generate_website_keywords,
    generate_heuristic_keywords,
)
from compass.plugin.one_shot.components import (
    SchemaBasedTextCollector,
    SchemaBasedTextExtractor,
    SchemaOrdinanceParser,
)
from compass.plugin.one_shot.cache import key_from_cache, key_to_cache
from compass.services.threaded import CLEANED_FP_REGISTRY
from compass.utilities.io import load_config
from compass.utilities.enums import LLMTasks
from compass.exceptions import COMPASSPluginConfigurationError


logger = logging.getLogger(__name__)
_SCHEMA_DIR = importlib.resources.files("compass.plugin.one_shot.schemas")
_QT_SEMAPHORE = Semaphore(1)
_WK_SEMAPHORE = Semaphore(1)
_HK_SEMAPHORE = Semaphore(1)


class _CacheKey(StrEnum):
    """LLM generated content cache keys"""

    QUERY_TEMPLATES = auto()
    WEBSITE_KEYWORDS = auto()
    HEURISTIC_KEYWORDS = auto()


def create_schema_based_one_shot_extraction_plugin(config, tech):  # noqa: C901
    """Create a one-shot extraction plugin based on a configuration

    Parameters
    ----------
    config : dict or path-like
        One-shot configuration dictionary. If not a dictionary, should
        be a path to a file containing the configuration (supported
        formats: JSON, JSON5, YAML, TOML). See the
        `wind ordinance schema <https://github.com/NatLabRockies/COMPASS/blob/main/examples/one_shot_schema_extraction/wind_schema.json>`_
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
            - `website_keywords`: A dictionary mapping keywords to
              scores for filtering websites during document retrieval.
              If not provided, the LLM will be used to generate
              website keywords based on the schema input.
            - `heuristic_keywords`: A dictionary containing the keyword
              lists used by the heuristic document filter. The
              dictionary must include ``not_tech_words``,
              ``good_tech_keywords``, ``good_tech_acronyms``, and
              ``good_tech_phrases`` keys. Alternatively, this input can
              simply be ``True``, in which case the LLM will be used to
              generate heuristic keyword lists based on the schema
              input. If ``False``, ``None``, or not provided, a `NoOp`
              heuristic that always returns ``True`` will be used (not
              recommended if doing website crawling).
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
            - `cache_llm_generated_content`: Boolean flag indicating
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
            - `doc_selection_method`: String defining the multi-doc
              selection option. Specifically, if multiple documents pass
              the filter, this method determines how the documents are
              submitted to the extraction context. Allowed options are:

                - "single doc": Use the first document that returns some
                  extracted data as the source document for the
                  extraction context.
                - "multi doc context": Submit text from multiple
                  documents to the extraction context simultaneously.
                - "multi doc all": Each document is extracted separately
                  and the results concatenated. This may give duplicated
                  feature results if the same feature is mentioned in
                  multiple documents.
                - "multi doc mixed": Each document is extracted
                  separately and the results are merged together at the
                  end. In this approach, each feature is reported at
                  most once.

              By default, ``"single doc"``.

    tech : str
        Technology identifier to use for the plugin (e.g., "wind",
        "solar"). Must be unique from the identifiers of any existing
        plugins.

    Returns
    -------
    callable
        A `SchemaBasedExtractionPlugin` subclass configured according to
        the input configuration.
    """
    if not isinstance(config, dict):
        config = load_config(config)

    if isinstance(config["schema"], str):
        config["schema"] = load_config(config["schema"])

    config["qual_feats"] = {
        f.casefold() for f in config["schema"].pop("$qualitative_features", [])
    }
    text_collectors = _collectors_from_config(config)
    text_extractors = _extractors_from_config(
        config, in_label=text_collectors[-1].OUT_LABEL, tech=tech
    )
    parsers = _parser_from_config(
        config, in_label=text_extractors[-1].OUT_LABEL
    )

    class SchemaBasedExtractionPlugin(OrdinanceExtractionPlugin):
        SCHEMA = config["schema"]
        """dict: Schema for the output of the text extraction step"""

        DOC_SELECTION_METHOD = DocSelectionMethod.normalize(
            config.get("doc_selection_method", "single doc")
        )
        """str: Method for selecting documents for extraction context

        Allowed options:

            - "single doc": Use the first document that returns some
              extracted data as the source document for the extraction
              context.
            - "multi doc context": Submit text from multiple documents
              to the extraction context simultaneously.
            - "multi doc all": Each document is extracted separately
              and the results concatenated. This may give duplicated
              feature results if the same feature is mentioned in
              multiple documents.
            - "multi doc mixed": Each document is extracted separately
              and the results are merged together at the end. In this
              approach, each feature is reported at most once.

        """

        IDENTIFIER = tech
        """str: Identifier for extraction task """

        HEURISTIC = NoOpHeuristic
        """BaseHeuristic: Class with a ``check()`` method"""

        HEURISTIC_KEYWORDS = None
        """dict: Keyword lists for heuristic content filtering"""

        TEXT_COLLECTORS = text_collectors
        """Classes for collecting text chunks from docs"""

        TEXT_EXTRACTORS = text_extractors
        """Classes for extracting cleaned text from collected text"""

        PARSERS = parsers
        """Classes for parsing structured ordinance data from text"""

        QUERY_TEMPLATES = []  # set by user or LLM-generated
        """list: List of search engine query templates"""

        WEBSITE_KEYWORDS = {}  # set by user or LLM-generated
        """dict: Keyword weight mapping for link crawl prioritization"""

        async def get_heuristic(self):
            """Get a `BaseHeuristic` instance with a `check()` method

            The ``check()`` method should accept a string of text and
            return ``True`` if the text passes the heuristic check and
            ``False`` otherwise.
            """
            if self.HEURISTIC_KEYWORDS and self.HEURISTIC is not NoOpHeuristic:
                return self.HEURISTIC()

            if not config.get("heuristic_keywords"):
                return NoOpHeuristic()

            hk = await self._get_heuristic_keywords()

            class SchemaBasedHeuristic(KeywordBasedHeuristic):
                NOT_TECH_WORDS = hk["NOT_TECH_WORDS"]
                GOOD_TECH_KEYWORDS = hk["GOOD_TECH_KEYWORDS"]
                GOOD_TECH_ACRONYMS = hk["GOOD_TECH_ACRONYMS"]
                GOOD_TECH_PHRASES = hk["GOOD_TECH_PHRASES"]

            self.__class__.HEURISTIC_KEYWORDS = hk
            self.__class__.HEURISTIC = SchemaBasedHeuristic
            return self.HEURISTIC()

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
                self.__class__.QUERY_TEMPLATES = qt
                return qt

            qt = key_from_cache(
                self.IDENTIFIER,
                config["schema"],
                key=_CacheKey.QUERY_TEMPLATES,
            )
            if qt:
                self.__class__.QUERY_TEMPLATES = qt
                return qt

            async with _QT_SEMAPHORE:
                if self.QUERY_TEMPLATES:
                    return self.QUERY_TEMPLATES

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
                logger.debug(
                    "Generated the following query templates:\n%r", qt
                )
                self.__class__.QUERY_TEMPLATES = qt

                if config.get("cache_llm_generated_content", True):
                    key_to_cache(
                        self.IDENTIFIER,
                        config["schema"],
                        key=_CacheKey.QUERY_TEMPLATES,
                        value=qt,
                    )

            return qt

        async def get_website_keywords(self):
            """Get a dict of website search keyword scores

            Returns
            -------
            dict
                Dictionary mapping keywords to scores that indicate
                links which should be prioritized when performing a
                website scrape for a document.
            """
            if self.WEBSITE_KEYWORDS:
                return self.WEBSITE_KEYWORDS

            if wk := config.get("website_keywords"):
                wk = _augment_website_keywords(wk)
                self.__class__.WEBSITE_KEYWORDS = wk
                return wk

            wk = key_from_cache(
                self.IDENTIFIER,
                config["schema"],
                key=_CacheKey.WEBSITE_KEYWORDS,
            )
            if wk:
                wk = _augment_website_keywords(wk)
                self.__class__.WEBSITE_KEYWORDS = wk
                return wk

            async with _WK_SEMAPHORE:
                if self.WEBSITE_KEYWORDS:
                    return self.WEBSITE_KEYWORDS

                model_config = self.model_configs.get(
                    LLMTasks.PLUGIN_GENERATION,
                    self.model_configs[LLMTasks.DEFAULT],
                )
                schema_llm = SchemaOutputLLMCaller(
                    llm_service=model_config.llm_service,
                    usage_tracker=self.usage_tracker,
                    **model_config.llm_call_kwargs,
                )
                logger.debug("Generating website keywords...")
                wk = await generate_website_keywords(
                    schema_llm,
                    config["schema"],
                    add_think_prompt=True,
                )
                logger.debug(
                    "Generated the following website keywords:\n%r", wk
                )
                if config.get("cache_llm_generated_content", True):
                    key_to_cache(
                        self.IDENTIFIER,
                        config["schema"],
                        key=_CacheKey.WEBSITE_KEYWORDS,
                        value=wk,
                    )

                wk = _augment_website_keywords(wk)
                self.__class__.WEBSITE_KEYWORDS = wk

            return wk

        async def _get_heuristic_keywords(self):
            """Get keyword lists for the heuristic document filter"""
            if self.HEURISTIC_KEYWORDS:
                return self.HEURISTIC_KEYWORDS

            if isinstance(hk := config.get("heuristic_keywords"), dict):
                hk = _normalize_heuristic_keywords(hk)
                self.__class__.HEURISTIC_KEYWORDS = hk
                return hk

            hk = key_from_cache(
                self.IDENTIFIER,
                config["schema"],
                key=_CacheKey.HEURISTIC_KEYWORDS,
            )
            if hk:
                hk = _normalize_heuristic_keywords(hk)
                self.__class__.HEURISTIC_KEYWORDS = hk
                return hk

            async with _HK_SEMAPHORE:
                if self.HEURISTIC_KEYWORDS:
                    return self.HEURISTIC_KEYWORDS

                model_config = self.model_configs.get(
                    LLMTasks.PLUGIN_GENERATION,
                    self.model_configs[LLMTasks.DEFAULT],
                )
                schema_llm = SchemaOutputLLMCaller(
                    llm_service=model_config.llm_service,
                    usage_tracker=self.usage_tracker,
                    **model_config.llm_call_kwargs,
                )
                logger.debug("Generating heuristic keywords...")
                hk = await generate_heuristic_keywords(
                    schema_llm,
                    config["schema"],
                    add_think_prompt=True,
                )
                hk = _normalize_heuristic_keywords(hk)
                logger.debug(
                    "Generated the following heuristic keywords:\n%r", hk
                )
                if config.get("cache_llm_generated_content", True):
                    key_to_cache(
                        self.IDENTIFIER,
                        config["schema"],
                        key=_CacheKey.HEURISTIC_KEYWORDS,
                        value=hk,
                    )

                self.__class__.HEURISTIC_KEYWORDS = hk

            return hk

        def _validate_query_templates(self):
            """NoOp validation for query templates

            Since templates can be generated by LLM, we don't know until
            runtime whether or not they will be valid.
            """

        def _validate_website_keywords(self):
            """NoOp validation for website keywords

            Since keywords can be generated by LLM, we don't know until
            runtime whether or not they will be valid.
            """

    register_plugin(SchemaBasedExtractionPlugin)
    return SchemaBasedExtractionPlugin


def _collectors_from_config(config):
    """Create a TextCollector subclass based on a config dict"""
    cp = config.get("collection_prompts")

    if cp is True:
        schema_fp = _SCHEMA_DIR / "validate_chunk.json5"

        class PluginTextCollector(SchemaBasedTextCollector):
            OUT_LABEL = NoOpTextCollector.OUT_LABEL  # reuse label
            SCHEMA = config["schema"]
            OUTPUT_SCHEMA = load_config(schema_fp)

        return [PluginTextCollector]

    if cp:

        class PluginTextCollector(PromptBasedTextCollector):
            OUT_LABEL = NoOpTextCollector.OUT_LABEL  # reuse label
            PROMPTS = cp

        return [PluginTextCollector]

    return [NoOpTextCollector]


def _extractors_from_config(config, in_label, tech):
    """Create a TextExtractor subclass based on a config dict"""
    tep = config.get("text_extraction_prompts")

    if tep is True:
        schema_fp = _SCHEMA_DIR / "extract_text.json5"

        class PluginTextExtractor(SchemaBasedTextExtractor):
            IN_LABEL = in_label
            OUT_LABEL = "copied_relevant_text"
            SCHEMA = config["schema"]
            OUTPUT_SCHEMA = load_config(schema_fp)

        CLEANED_FP_REGISTRY.setdefault(tech.casefold(), {})[
            "copied_relevant_text"
        ] = "Text for Extraction.txt"
        return [PluginTextExtractor]

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
        QUALITATIVE_FEATURES = config["qual_feats"]
        DATA_TYPE_SHORT_DESC = config.get("data_type_short_desc")
        SYSTEM_PROMPT = new_sys_prompt

    return [PluginParser]


def _augment_website_keywords(keywords):
    """Add URL-encoded variants for multi-word keywords"""
    augmented = dict(keywords)
    for keyword, score in list(augmented.items()):
        if not isinstance(keyword, str):
            continue

        if " " not in keyword:
            continue

        encoded = keyword.replace(" ", "%20")
        if encoded not in augmented:
            augmented[encoded] = score

        plus_encoded = keyword.replace(" ", "+")
        if plus_encoded not in augmented:
            augmented[plus_encoded] = score

    return augmented


def _normalize_heuristic_keywords(raw):
    """Normalize heuristic keyword lists into required structure"""
    if not isinstance(raw, dict):
        msg = "Heuristic keywords must be a dictionary of keyword lists."
        raise COMPASSPluginConfigurationError(msg)

    expected_keys = {
        "NOT_TECH_WORDS",
        "GOOD_TECH_KEYWORDS",
        "GOOD_TECH_ACRONYMS",
        "GOOD_TECH_PHRASES",
    }

    normalized = {}
    for raw_key, value in raw.items():
        if not isinstance(raw_key, str):
            msg = "Heuristic keyword keys must be strings."
            raise COMPASSPluginConfigurationError(msg)

        target_key = (
            raw_key.strip().replace(" ", "_").replace("-", "_").upper()
        )
        if target_key not in expected_keys:
            msg = f"Unexpected heuristic keyword list: {raw_key!r}."
            raise COMPASSPluginConfigurationError(msg)

        normalized[target_key] = _normalize_keyword_list(value)

    missing = expected_keys - set(normalized)
    if missing:
        msg = (
            f"Heuristic keywords are missing required lists: {sorted(missing)}"
        )
        raise COMPASSPluginConfigurationError(msg)

    empty = [key for key, value in normalized.items() if not value]
    if empty:
        msg = f"Heuristic keyword lists must not be empty: {sorted(empty)}"
        raise COMPASSPluginConfigurationError(msg)

    return normalized


def _normalize_keyword_list(items):
    """Normalize keyword list entries"""
    normalized = set()
    for item in items:
        if not isinstance(item, str):
            continue

        keyword = item.strip()
        if not keyword:
            continue

        keyword = keyword.casefold()
        if keyword in normalized:
            continue

        normalized.add(keyword)

    return list(normalized)
