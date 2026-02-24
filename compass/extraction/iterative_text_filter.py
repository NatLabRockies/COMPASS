"""Text filtering for focused re-extraction"""

import asyncio
import json
import re
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from compass.llm.calling import SchemaOutputLLMCaller
from compass.utilities.enums import LLMUsageCategory


logger = logging.getLogger(__name__)


_KEYWORD_EXCLUDE = {
    "the",
    "and",
    "for",
    "from",
    "that",
    "this",
    "with",
}


_SEMANTIC_SYSTEM_MESSAGE = """\
You are a precision extraction relevance classifier.\
"""

_SEMANTIC_PROMPT_TEMPLATE = """\
Determine whether the chunk contains ordinance information relevant to\
the target feature.

# TARGET FEATURE #
Feature ID: {feature}
Feature context:
{feature_context}

# CHUNK TEXT #
{chunk}

# TASK #
Return:
- is_relevant: true if the chunk includes requirements, thresholds,\
  exceptions, procedures, or definitions that help extract the target\
  feature. Otherwise false.
- confidence: a number from 0.0 to 1.0 representing certainty.
- reason: brief explanation under 30 words.

Be conservative and return false when relevance is unclear.\
"""

_SEMANTIC_RELEVANCE_SCHEMA = {
    "type": "object",
    "description": "Chunk relevance classification",
    "additionalProperties": False,
    "required": ["is_relevant", "confidence", "reason"],
    "properties": {
        "is_relevant": {
            "type": "boolean",
            "description": "Whether this chunk is relevant",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Model confidence for relevance label",
        },
        "reason": {
            "type": "string",
            "description": "Short reason for the relevance decision",
        },
    },
}


class FocusedTextFilter:
    """Filter extraction text

    Filter ordinance text to focus on sections most likely to be
    relevant for specific features, using keyword and/or semantic
    search. This can improve extraction accuracy and efficiency by
    reducing irrelevant content. The context_window parameter allows
    including surrounding text for better understanding of relevant
    sections
    """

    def __init__(
        self,
        strategy="hybrid",
        context_window=2,
        llm_service=None,
        usage_tracker=None,
        **llm_kwargs,
    ):
        """

        Parameters
        ----------
        strategy : str, default="hybrid"
            Filtering strategy: "semantic", "keyword", or "hybrid"
        context_window : int, default=2
            Number of surrounding chunks to include for context
        llm_service : compass.services.base.Service, optional
            LLM service used for semantic relevance calls. By default,
            ``None``.
        usage_tracker : UsageTracker, optional
            Optional tracker for token usage during semantic relevance
            calls. By default, ``None``.
        **llm_kwargs
            Keyword arguments passed to semantic LLM calls.
        """
        self.strategy = strategy
        self.context_window = context_window
        self.llm_service = llm_service
        self.usage_tracker = usage_tracker
        self.llm_kwargs = llm_kwargs
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=200
        )
        self._semantic_caller = None
        if self.llm_service is not None:
            self._semantic_caller = SchemaOutputLLMCaller(
                llm_service=self.llm_service,
                usage_tracker=self.usage_tracker,
                **self.llm_kwargs,
            )

    async def filter_for_features(self, text, feature_list, schema):
        """Extract text sections relevant to specific features

        Parameters
        ----------
        text : str
            Full ordinance text
        feature_list : list of str
            Feature IDs to extract text for
        schema : dict
            Schema with feature descriptions for context

        Returns
        -------
        dict
            Mapping {feature_id: filtered_text}
        """
        chunks = self._splitter.split_text(text)
        logger.debug("Split text into %d chunks for filtering", len(chunks))

        filtered_texts = {}
        for feature in feature_list:
            if self.strategy == "semantic":
                indices = await self._semantic_search(chunks, feature, schema)
            elif self.strategy == "keyword":
                indices = _keyword_search(chunks, feature, schema)
            else:
                sem_indices = await self._semantic_search(
                    chunks, feature, schema
                )
                kw_indices = _keyword_search(chunks, feature, schema)
                indices = list(set(sem_indices + kw_indices))

            if not indices:
                logger.warning(
                    "No relevant chunks found for feature %r", feature
                )
                filtered_texts[feature] = text
                continue

            expanded_indices = self._expand_with_context(indices, len(chunks))

            filtered_texts[feature] = "\n\n".join(
                chunks[i] for i in sorted(expanded_indices)
            )

            logger.debug(
                "Filtered text for %r: %d chunks selected, %d with context",
                feature,
                len(indices),
                len(expanded_indices),
            )

        return filtered_texts

    async def _semantic_search(self, chunks, feature, schema):
        """Search for relevant chunks using semantic LLM calls"""
        if self._semantic_caller is None:
            logger.warning(
                "Semantic strategy requested without llm_service for %r; "
                "falling back to keyword",
                feature,
            )
            return _keyword_search(chunks, feature, schema)

        feature_context = _format_feature_context(feature, schema)
        tasks = [
            self._check_chunk_relevance(feature, feature_context, chunk)
            for chunk in chunks
        ]

        try:
            responses = await asyncio.gather(*tasks)
        except Exception:
            logger.exception(
                "Semantic search failed for %r, falling back to keyword",
                feature,
            )
            return _keyword_search(chunks, feature, schema)

        if any("is_relevant" not in response for response in responses):
            logger.warning(
                "Semantic search returned malformed response for %r; "
                "falling back to keyword",
                feature,
            )
            return _keyword_search(chunks, feature, schema)

        indices = [
            idx
            for idx, response in enumerate(responses)
            if response.get("is_relevant")
        ]

        logger.debug(
            "Semantic search for %r found %d matching chunks",
            feature,
            len(indices),
        )
        return indices

    async def _check_chunk_relevance(self, feature, feature_context, chunk):
        """Check if a text chunk is relevant to the target feature"""
        prompt = _SEMANTIC_PROMPT_TEMPLATE.format(
            feature=feature,
            feature_context=feature_context,
            chunk=chunk,
        )
        return await self._semantic_caller.call(
            sys_msg=_SEMANTIC_SYSTEM_MESSAGE,
            content=prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "chunk_relevance",
                    "strict": True,
                    "schema": _SEMANTIC_RELEVANCE_SCHEMA,
                },
            },
            usage_sub_label=LLMUsageCategory.DOCUMENT_CONTENT_VALIDATION,
        )

    def _expand_with_context(self, indices, total_chunks):
        """Expand selected chunks to include surrounding context"""
        expanded = set()
        for idx in indices:
            start = max(0, idx - self.context_window)
            end = min(total_chunks, idx + self.context_window + 1)
            expanded.update(range(start, end))
        return expanded


def _keyword_search(chunks, feature, schema):
    """Find chunk indices that match keywords for the feature"""
    keywords = _extract_keywords(feature, schema)
    matching_indices = []

    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        if any(kw.lower() in chunk_lower for kw in keywords):
            matching_indices.append(i)

    logger.debug(
        "Keyword search for %r found %d matching chunks",
        feature,
        len(matching_indices),
    )
    return matching_indices


def _extract_keywords(feature, schema):
    """Extract keywords from feature name and schema description"""
    keywords = set()

    feature_parts = re.split(r"[_\-\s]+", feature.lower())
    keywords.update(feature_parts)

    if isinstance(schema, dict):
        feature_info = schema.get(feature, {})

        if isinstance(feature_info, dict):
            description = feature_info.get("description", "")
        elif isinstance(feature_info, str):
            description = feature_info
        else:
            description = ""

        if description:
            words = re.findall(r"\b[a-zA-Z]{4,}\b", description.lower())
            keywords.update(words[:10])

    keywords -= _KEYWORD_EXCLUDE

    logger.debug(
        "Extracted keywords for %r: %s", feature, ", ".join(list(keywords)[:5])
    )

    return list(keywords)


def _format_feature_context(feature, schema):
    """Format feature-specific schema context for semantic prompts"""
    feature_info = {}
    if isinstance(schema, dict):
        feature_info = schema.get(feature, {})

    if isinstance(feature_info, dict):
        return json.dumps(feature_info, indent=2)
    if isinstance(feature_info, str):
        return feature_info
    return "No feature context provided"
