"""Ordinance document content Validation logic

These are primarily used to validate that a legal document applies to a
particular technology (e.g. Large Wind Energy Conversion Systems).
"""

import logging


logger = logging.getLogger(__name__)


class ValidationWithMemory:
    """Validate text chunks by sometimes looking at previous chunks"""

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
                    usage_sub_label="document_content_validation",
                )
                check = mem[key] = content.get(key, False)
            if check:
                return check
        return False
