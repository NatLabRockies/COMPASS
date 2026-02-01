"""Water ordinance document content collection and extraction

These methods help filter down the document text to only the portions
relevant to water rights ordinances.
"""

import logging

from compass.common import BaseTextExtractor
from compass.llm.calling import StructuredLLMCaller
from compass.utilities.parsing import merge_overlapping_texts
from compass.utilities.enums import LLMUsageCategory


logger = logging.getLogger(__name__)


class WaterRightsHeuristic:
    """NoOp heuristic check"""

    def check(self, *__, **___):  # noqa: PLR6301
        """Always return ``True`` for water rights documents"""
        return True


class WaterRightsTextCollector(StructuredLLMCaller):
    """Check text chunks for ordinances and collect them if they do"""

    LABEL = "relevant_text"
    """Identifier for text collected by this class"""

    WELL_PERMITS_PROMPT = (
        "You extract structured data from text. Return your answer in JSON "
        "format (not markdown). Your JSON file must include exactly three "
        "keys. The first key is 'district_rules' which is a string summarizes "
        "the rules associated with the groundwater conservation district. "
        "The second key is 'well_requirements', which is a string that "
        "summarizes the requirements for drilling a groundwater well. The "
        "last key is '{key}', which is a boolean that is set to True if the "
        "text excerpt provides substantive information related to the "
        "groundwater conservation district's rules or management plans. "
    )
    """Prompt to check if chunk contains water rights ordinance info"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chunks = {}

    @property
    def relevant_text(self):
        """str: Combined ordinance text from the individual chunks"""
        if not self._chunks:
            logger.debug(
                "No relevant water rights chunk(s) found in original text",
            )
            return ""

        logger.debug(
            "Grabbing %d water rights chunk(s) from original text at these "
            "indices: %s",
            len(self._chunks),
            list(self._chunks),
        )

        text = [self._chunks[ind] for ind in sorted(self._chunks)]
        return merge_overlapping_texts(text)

    async def check_chunk(self, chunk_parser, ind):
        """Check a chunk at a given ind to see if it contains ordinance

        Parameters
        ----------
        chunk_parser : ParseChunksWithMemory
            Instance that contains a ``parse_from_ind`` method.
        ind : int
            Index of the chunk to check.

        Returns
        -------
        bool
            Boolean flag indicating whether or not the text in the chunk
            contains water rights ordinance text.
        """
        contains_ord_info = await chunk_parser.parse_from_ind(
            ind,
            key="contains_ord_info",
            llm_call_callback=self._check_chunk_contains_ord,
        )

        if contains_ord_info:
            logger.debug(
                "Text at ind %d contains water rights ordinance info", ind
            )
            _store_chunk(chunk_parser, ind, self._chunks)
        else:
            logger.debug(
                "Text at ind %d does not contain water rights ordinance info",
                ind,
            )

        return contains_ord_info

    async def _check_chunk_contains_ord(self, key, text_chunk):
        """Call LLM on a chunk of text to check for ordinance"""
        content = await self.call(
            sys_msg=self.WELL_PERMITS_PROMPT.format(key=key),
            content=text_chunk,
            usage_sub_label=(LLMUsageCategory.DOCUMENT_CONTENT_VALIDATION),
        )
        logger.debug("LLM response: %s", content)
        return content.get(key, False)


class WaterRightsTextExtractor(BaseTextExtractor):
    """No-Op text extractor"""

    @property
    def parsers(self):
        """Iterable of parsers provided by this extractor

        Yields
        ------
        name : str
            Name describing the type of text output by the parser.
        parser : callable
            Async function that takes a ``text_chunks`` input and
            outputs parsed text.
        """
        yield "cleaned_text_for_extraction", merge_overlapping_texts


def _store_chunk(parser, chunk_ind, store):
    """Store chunk and its neighbors if it is not already stored"""
    for offset in range(1 - parser.num_to_recall, 2):
        ind_to_grab = chunk_ind + offset
        if ind_to_grab < 0 or ind_to_grab >= len(parser.text_chunks):
            continue

        store.setdefault(ind_to_grab, parser.text_chunks[ind_to_grab])
