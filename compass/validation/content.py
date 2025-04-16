"""Ordinance document content Validation logic

These are primarily used to validate that a legal document applies to a
particular technology (e.g. Large Wind Energy Conversion Systems).
"""

import asyncio
import logging
from abc import ABC, abstractmethod

from compass.utilities.enums import LLMUsageCategory


logger = logging.getLogger(__name__)


class ParseChunksWithMemory:
    """Check text chunks by sometimes looking at previous chunks

    The idea behind this approach is that sometimes the context for a
    setback or other ordinances is found in a previous chunk, so it may
    be worthwhile (especially for validation purposes) to check a few
    text chunks back for some validation pieces. In order to do this
    semi-efficiently, we make use of a cache that's labeled "memory".
    """

    def __init__(self, structured_llm_caller, text_chunks, num_to_recall=2):
        """

        Parameters
        ----------
        structured_llm_caller : compass.llm.StructuredLLMCaller
            StructuredLLMCaller instance. Used for structured validation
            queries.
        text_chunks : list of str
            List of strings, each of which represent a chunk of text.
            The order of the strings should be the order of the text
            chunks. This validator may refer to previous text chunks to
            answer validation questions.
        num_to_recall : int, optional
            Number of chunks to check for each validation call. This
            includes the original chunk! For example, if
            `num_to_recall=2`, the validator will first check the chunk
            at the requested index, and then the previous chunk as well.
            By default, ``2``.
        """
        self.slc = structured_llm_caller
        self.text_chunks = text_chunks
        self.num_to_recall = num_to_recall
        self.memory = [{} for _ in text_chunks]

    # fmt: off
    def _inverted_mem(self, starting_ind):
        """Inverted memory"""
        inverted_mem = self.memory[:starting_ind + 1:][::-1]
        yield from inverted_mem[:self.num_to_recall]

    # fmt: off
    def _inverted_text(self, starting_ind):
        """Inverted text chunks"""
        inverted_text = self.text_chunks[:starting_ind + 1:][::-1]
        yield from inverted_text[:self.num_to_recall]

    async def parse_from_ind(self, ind, prompt, key):
        """Validate a chunk of text

        Validation occurs by querying the LLM using the input prompt and
        parsing the `key` from the response JSON. The prompt should
        request that the key be a boolean output. If the key retrieved
        from the LLM response is False, a number of previous text chunks
        are checked as well, using the same prompt. This can be helpful
        in cases where the answer to the validation prompt (e.g. does
        this text pertain to a large WECS?) is only found in a previous
        text chunk.

        Parameters
        ----------
        ind : int
            Positive integer corresponding to the chunk index.
            Must be less than `len(text_chunks)`.
        prompt : str
            Input LLM system prompt that describes the validation
            question. This should request a JSON output from the LLM.
            It should also take `key` as a formatting input.
        key : str
            A key expected in the JSON output of the LLM containing the
            response for the validation question. This string will also
            be used to format the system prompt before it is passed to
            the LLM.

        Returns
        -------
        bool
            ``True`` if the LLM returned ``True`` for this text chunk or
            `num_to_recall-1` text chunks before it.
            ``False`` otherwise.
        """
        logger.debug("Checking %r for ind %d", key, ind)
        mem_text = zip(
            self._inverted_mem(ind), self._inverted_text(ind), strict=False
        )
        for step, (mem, text) in enumerate(mem_text):
            logger.debug("Mem at ind %d is %s", step, mem)
            check = mem.get(key)
            if check is None:
                content = await self.slc.call(
                    sys_msg=prompt.format(key=key),
                    content=text,
                    usage_sub_label=(
                        LLMUsageCategory.DOCUMENT_CONTENT_VALIDATION
                    ),
                )
                logger.debug("LLM response: %s", str(content))
                check = mem[key] = content.get(key, False)
            if check:
                return check
        return False


class Heuristic(ABC):
    """Perform a heuristic check for mention of a technology in text"""

    _GOOD_ACRONYM_CONTEXTS = [
        " {acronym} ",
        " {acronym}\n",
        " {acronym}.",
        "\n{acronym} ",
        "\n{acronym}.",
        "\n{acronym}\n",
        "({acronym} ",
        " {acronym})",
    ]

    def check(self, text, match_count_threshold=1):
        """Check for mention of a tech in text

        This check first strips the text of any tech "look-alike" words
        (e.g. "window", "windshield", etc for "wind" technology). Then,
        it checks for particular keywords, acronyms, and phrases that
        pertain to the tech in the text. If enough keywords are mentions
        (as dictated by `match_count_threshold`), this check returns
        ``True``.

        Parameters
        ----------
        text : str
            Input text that may or may not mention the technology of
            interest.
        match_count_threshold : int, optional
            Number of keywords that must match for the text to pass this
            heuristic check. Count must be strictly greater than this
            value. By default, ``1``.

        Returns
        -------
        bool
            ``True`` if the number of keywords/acronyms/phrases detected
            exceeds the `match_count_threshold`.
        """
        heuristics_text = self._convert_to_heuristics_text(text)
        total_keyword_matches = self._count_single_keyword_matches(
            heuristics_text
        )
        total_keyword_matches += self._count_acronym_matches(heuristics_text)
        total_keyword_matches += self._count_phrase_matches(heuristics_text)
        return total_keyword_matches > match_count_threshold

    def _convert_to_heuristics_text(self, text):
        """Convert text for heuristic content parsing"""
        heuristics_text = text.casefold()
        for word in self.NOT_TECH_WORDS:
            heuristics_text = heuristics_text.replace(word, "")
        return heuristics_text

    def _count_single_keyword_matches(self, heuristics_text):
        """Count number of good tech keywords that appear in text"""
        return sum(
            keyword in heuristics_text for keyword in self.GOOD_TECH_KEYWORDS
        )

    def _count_acronym_matches(self, heuristics_text):
        """Count number of good tech acronyms that appear in text"""
        acronym_matches = 0
        for context in self._GOOD_ACRONYM_CONTEXTS:
            acronym_keywords = {
                context.format(acronym=acronym)
                for acronym in self.GOOD_TECH_ACRONYMS
            }
            acronym_matches = sum(
                keyword in heuristics_text for keyword in acronym_keywords
            )
            if acronym_matches > 0:
                break
        return acronym_matches

    def _count_phrase_matches(self, heuristics_text):
        """Count number of good tech phrases that appear in text"""
        return sum(
            all(keyword in heuristics_text for keyword in phrase.split(" "))
            for phrase in self.GOOD_TECH_PHRASES
        )

    @property
    @abstractmethod
    def NOT_TECH_WORDS(self):  # noqa: N802
        """iter: Iterable of words that don't pertain to the tech"""
        raise NotImplementedError

    @property
    @abstractmethod
    def GOOD_TECH_KEYWORDS(self):  # noqa: N802
        """iter: Iterable of keywords that pertain to the tech"""
        raise NotImplementedError

    @property
    @abstractmethod
    def GOOD_TECH_ACRONYMS(self):  # noqa: N802
        """iter: Iterable of acronyms that pertain to the tech"""
        raise NotImplementedError

    @property
    @abstractmethod
    def GOOD_TECH_PHRASES(self):  # noqa: N802
        """iter: Iterable of phrases that pertain to the tech"""
        raise NotImplementedError


class LegalTextValidator:
    """Parse chunks to determine if they contain legal text"""

    IS_LEGAL_TEXT_PROMPT = (
        "# CONTEXT #\n"
        "You are an AI designed to classify text excerpts based on their "
        "source type. The goal is to identify **legally binding regulations "
        "(such as zoning ordinances or enforceable bans)** and filter out "
        "text that does not belong to an in-effect legal statute.\n"
        "Legally binding regulations are formal laws enacted by a governing "
        "authority, containing enforceable rules or restrictions. However, "
        "many documents discuss, summarize, or propose zoning rules without "
        "having legal force. Examples of non-binding documents include, but "
        "are not limited to, **news articles, reports, draft ordinances, "
        "model ordinances, public notices, surveys, development plans, "
        "project-specific plans, conservation plans, summaries, etc.**\n"
        "\n# OBJECTIVE #\n"
        "Your task is to analyze a given text excerpt and determine whether "
        "it belongs to a **legally binding regulation.** If the text is from "
        "any other type of document, it should be classified accordingly and "
        "excluded from legally binding regulations.\n"
        "\n# RESPONSE #\n"
        "Return the classification in **JSON format** with exactly three "
        "keys:\n\n"
        '1. **"summary"** (string) - A concise summary of the text.\n'
        '2. **"type"** (string) - The best-fitting category of the text.\n'
        '3. **"{key}"** (boolean) -\n'
        "\t- `true` if the text is a **legally binding regulation**.\n"
        "\t- `false` if the text belongs to any other type of document.\n\n"
        "Ensure precise classification by prioritizing **legal "
        "enforceability** over content similarity."
    )

    def __init__(self):
        self._legal_text_mem = []

    @property
    def is_legal_text(self):
        """bool: ``True`` if text was found to be from a legal source"""
        if not self._legal_text_mem:
            return False
        return sum(self._legal_text_mem) >= 0.5 * len(self._legal_text_mem)

    async def check_chunk(self, chunk_parser, ind):
        """Check a chunk at a given ind to see if it contains legal text

        Parameters
        ----------
        chunk_parser : ParseChunksWithMemory
            Instance of `ParseChunksWithMemory` that contains a
            `parse_from_ind` method.
        ind : int
            Index of the chunk to check.

        Returns
        -------
        bool
            Boolean flag indicating whether or not the text in the chunk
            resembles legal text.
        """
        is_legal_text = await chunk_parser.parse_from_ind(
            ind, self.IS_LEGAL_TEXT_PROMPT, key="legal_text"
        )
        self._legal_text_mem.append(is_legal_text)
        if is_legal_text:
            logger.debug("Text at ind %d is legal text", ind)
        else:
            logger.debug("Text at ind %d is not legal text", ind)
        return is_legal_text


async def parse_by_chunks(
    chunk_parser,
    heuristic,
    legal_text_validator,
    callbacks=None,
    min_chunks_to_process=3,
):
    """Parse text by chunks, passing to callbacks if it's legal text

    This method goes through the chunks one by one, and passes them to
    the callback parsers if the `legal_text_validator` check passes. If
    `min_chunks_to_process` number of chunks fail the legal text check,
    parsing is aborted.

    Parameters
    ----------
    chunk_parser : ParseChunksWithMemory
        Instance of `ParseChunksWithMemory` that contains the attributes
        `text_chunks` and `num_to_recall`. The chunks in the
        `text_chunks` attribute will be iterated over.
    heuristic : Heuristic
        Instance of `Heuristic` with a `check` method. This should be a
        fast check meant to quickly dispose of chunks of text. Any chunk
        that fails this check will NOT be passed to the callback
        parsers.
    legal_text_validator : LegalTextValidator
        Instance of `LegalTextValidator` that can be used to validate
        each chunk for legal text.
    callbacks : list, optional
        List of async callbacks that take a `chunk_parser` and `index`
        as inputs and return a boolean determining whether the text
        chunk was parsed successfully or not. By default, ``None``,
        which does not use any callbacks.
    min_chunks_to_process : int, optional
        Minimum number of chunks to process before aborting due to text
        not being legal. By default, ``3``.
    """
    passed_heuristic_mem = []
    callbacks = callbacks or []
    outer_task_name = asyncio.current_task().get_name()

    for ind, text in enumerate(chunk_parser.text_chunks):
        passed_heuristic_mem.append(heuristic.check(text))
        if ind < min_chunks_to_process:
            is_legal = await legal_text_validator.check_chunk(
                chunk_parser, ind
            )
            if not is_legal:  # don't bother checking this chunk
                continue

        # don't bother checking this document
        elif not legal_text_validator.is_legal_text:
            return

        # hasn't passed heuristic, so don't pass it to callbacks
        elif not any(passed_heuristic_mem[-chunk_parser.num_to_recall :]):
            continue

        logger.debug("Processing text at ind %d", ind)
        logger.debug_to_file("Text:\n%s", text)

        if not callbacks:
            continue

        cb_futures = [
            asyncio.create_task(cb(chunk_parser, ind), name=outer_task_name)
            for cb in callbacks
        ]
        cb_results = await asyncio.gather(*cb_futures)

        # mask this chunk if we got a good result - this avoids forcing
        # the following chunk to be checked (it will only be checked if
        # it itself passes the heuristic)
        passed_heuristic_mem[-1] = not any(cb_results)
