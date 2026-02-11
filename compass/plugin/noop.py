"""COMPASS NoOp plugin implementation"""

import logging

from compass.plugin.interface import BaseHeuristic, BaseTextCollector
from compass.plugin.ordinance import BaseTextExtractor
from compass.utilities.parsing import merge_overlapping_texts


logger = logging.getLogger(__name__)


class NoOpHeuristic(BaseHeuristic):
    """NoOp heuristic check"""

    def check(self, *__, **___):  # noqa: PLR6301
        """Always return ``True``"""
        return True


class NoOpTextCollector(BaseTextCollector):
    """NoOp text collector that returns the full text"""

    OUT_LABEL = "relevant_text"
    """Identifier for text collected by this class"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chunks = {}

    @property
    def relevant_text(self):
        """str: Combined relevant text from the individual chunks"""
        text = [self._chunks[ind] for ind in sorted(self._chunks)]
        return merge_overlapping_texts(text)

    async def check_chunk(self, chunk_parser, ind):
        """Check a chunk at a given ind to see if it contains ordinance

        In this implementation, we store all chunks, so this method
        always returns ``True``.

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
            contains large wind energy conversion system ordinance text.
        """
        logger.debug(
            "NoOpTextCollector: assuming all text is relevant, so adding "
            "chunk at ind %d to extraction text",
            ind,
        )
        self._store_chunk(chunk_parser, ind)
        return True

    def _store_chunk(self, parser, chunk_ind):
        """Store chunk and its neighbors if it is not already stored"""
        for offset in range(1 - parser.num_to_recall, 2):
            ind_to_grab = chunk_ind + offset
            if ind_to_grab < 0 or ind_to_grab >= len(parser.text_chunks):
                continue

            self._chunks.setdefault(
                ind_to_grab, parser.text_chunks[ind_to_grab]
            )


class NoOpTextExtractor(BaseTextExtractor):
    """NoOp text extractor that returns the full text"""

    def __init__(self, llm_caller):
        """

        Parameters
        ----------
        llm_caller : LLMCaller
            LLM Caller instance used to extract ordinance info with.
        """
        self.llm_caller = llm_caller

    async def return_original(self, text_chunks):  # noqa: PLR6301
        """No processing, just return original text

        Parameters
        ----------
        text_chunks : list of str
            List of strings, each of which represent a chunk of text.
            The order of the strings should be the order of the text
            chunks.

        Returns
        -------
        str
            Ordinance text extracted from text chunks.
        """
        logger.debug(
            "No text extraction prompts provided; returning original text"
        )
        return merge_overlapping_texts(text_chunks)

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
        yield self.OUT_LABEL, self.return_original
