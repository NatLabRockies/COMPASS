"""COMPASS Ordinance Location Validation logic

These are primarily used to validate that a legal document applies to a
particular location.
"""

import asyncio
import logging

from elm.web.file_loader import AsyncWebFileLoader

from compass.llm.calling import BaseLLMCaller, ChatLLMCaller, LLMCaller
from compass.common import setup_async_decision_tree, run_async_tree
from compass.validation.graphs import (
    setup_graph_correct_jurisdiction_type,
    setup_graph_correct_jurisdiction_from_url,
)
from compass.utilities.enums import LLMUsageCategory


logger = logging.getLogger(__name__)


class DTreeURLJurisdictionValidator(BaseLLMCaller):
    """Validator that checks whether a URL matches a jurisdiction"""

    SYSTEM_MESSAGE = (
        "You are an expert data analyst that examines URLs to determine if "
        "they contain information about jurisdictions. Only ever answer "
        "based on the information in the URL itself."
    )
    """System message for URL jurisdiction validation LLM calls"""

    def __init__(self, jurisdiction, **kwargs):
        """

        Parameters
        ----------
        structured_llm_caller : StructuredLLMCaller
            Instance used for structured validation queries.
        **kwargs
            Additional keyword arguments to pass to the
            :class:`~compass.llm.calling.BaseLLMCaller` instance.
        """
        super().__init__(**kwargs)
        self.jurisdiction = jurisdiction

    async def check(self, url):
        """Check if the content passes the validation

        Parameters
        ----------
        content : str
            Document content to validate.

        Returns
        -------
        bool
            ``True`` if the content passes the validation check,
            ``False`` otherwise.
        """
        if not url:
            return False

        chat_llm_caller = ChatLLMCaller(
            llm_service=self.llm_service,
            system_message=self.SYSTEM_MESSAGE,
            usage_tracker=self.usage_tracker,
            **self.kwargs,
        )
        tree = setup_async_decision_tree(
            setup_graph_correct_jurisdiction_from_url,
            usage_sub_label=LLMUsageCategory.URL_JURISDICTION_VALIDATION,
            jurisdiction=self.jurisdiction,
            url=url,
            chat_llm_caller=chat_llm_caller,
        )
        out = await run_async_tree(tree, response_as_json=True)
        return self._parse_output(out)

    def _parse_output(self, props):  # noqa: PLR6301
        """Parse LLM response and return boolean validation result"""
        logger.debug(
            "Parsing URL jurisdiction validation output:\n\t%s", props
        )
        return len(props) > 0 and all(props.values())


class DTreeJurisdictionValidator(BaseLLMCaller):
    """Jurisdiction Validation using a decision tree"""

    META_SCORE_KEY = "Jurisdiction Validation Score"
    """Key in doc.attrs where score is stored"""

    SYSTEM_MESSAGE = (
        "You are a legal expert assisting a user with determining the scope "
        "of applicability for their legal ordinance documents."
    )
    """System message for jurisdiction validation LLM calls"""

    def __init__(self, jurisdiction, **kwargs):
        """

        Parameters
        ----------
        structured_llm_caller : StructuredLLMCaller
            Instance used for structured validation queries.
        **kwargs
            Additional keyword arguments to pass to the
            :class:`~compass.llm.calling.BaseLLMCaller` instance.
        """
        super().__init__(**kwargs)
        self.jurisdiction = jurisdiction

    async def check(self, content):
        """Check if the content passes the validation

        Parameters
        ----------
        content : str
            Document content to validate.

        Returns
        -------
        bool
            ``True`` if the content passes the validation check,
            ``False`` otherwise.
        """
        if not content:
            return False

        chat_llm_caller = ChatLLMCaller(
            llm_service=self.llm_service,
            system_message=self.SYSTEM_MESSAGE,
            usage_tracker=self.usage_tracker,
            **self.kwargs,
        )
        tree = setup_async_decision_tree(
            setup_graph_correct_jurisdiction_type,
            usage_sub_label=LLMUsageCategory.DOCUMENT_JURISDICTION_VALIDATION,
            jurisdiction=self.jurisdiction,
            text=content,
            chat_llm_caller=chat_llm_caller,
        )
        out = await run_async_tree(tree, response_as_json=True)
        return self._parse_output(out)

    def _parse_output(self, props):  # noqa: PLR6301
        """Parse LLM response and return boolean validation result"""
        logger.debug(
            "Parsing county jurisdiction validation output:\n\t%s", props
        )
        return props.get("correct_jurisdiction")


class JurisdictionValidator:
    """COMPASS Ordinance Jurisdiction validator

    Combines the logic of several validators into a single class.

    Purpose:
        Determine whether a document pertains to a specific county.
    Responsibilities:
        1. Use a combination of heuristics and LLM queries to determine
           whether or not a document pertains to a particular county.
    Key Relationships:
        Uses a StructuredLLMCaller for LLM queries and delegates
        sub-validation to
        :class:`DTreeJurisdictionValidator`,
        and :class:`DTreeURLJurisdictionValidator`.
    """

    def __init__(self, score_thresh=0.8, text_splitter=None, **kwargs):
        """

        Parameters
        ----------
        score_thresh : float, optional
            Score threshold to exceed when voting on content from raw
            pages. By default, ``0.8``.
        text_splitter : LCTextSplitter, optional
            Optional text splitter instance to attach to doc (used for
            splitting out pages in an HTML document).
            By default, ``None``.
        **kwargs
            Additional keyword arguments to pass to the
            :class:`~compass.llm.calling.BaseLLMCaller` instance.
        """
        self.score_thresh = score_thresh
        self.text_splitter = text_splitter
        self.kwargs = kwargs

    async def check(self, doc, jurisdiction):
        """Check if the document belongs to the county

        Parameters
        ----------
        doc : elm.web.document.BaseDocument
            Document instance. Should contain a "source" key in the
            ``attrs`` that contains a URL (used for the URL validation
            check). Raw content will be parsed for county name and
            correct jurisdiction.

        Returns
        -------
        bool
            `True` if the doc contents pertain to the input county.
            `False` otherwise.
        """
        if hasattr(doc, "text_splitter") and self.text_splitter is not None:
            old_splitter = doc.text_splitter
            doc.text_splitter = self.text_splitter
            out = await self._check(doc, jurisdiction)
            doc.text_splitter = old_splitter
            return out

        return await self._check(doc, jurisdiction)

    async def _check(self, doc, jurisdiction):
        """Check if the document belongs to the county"""
        if self.text_splitter is not None:
            doc.text_splitter = self.text_splitter

        url = doc.attrs.get("source")
        if url:
            logger.debug("Checking URL (%s) for jurisdiction name...", url)
            url_validator = DTreeURLJurisdictionValidator(
                jurisdiction, **self.kwargs
            )
            url_is_correct_jurisdiction = await url_validator.check(url)
            if url_is_correct_jurisdiction:
                return True

        logger.info("Validating document from source: %s", url or "Unknown")
        logger.debug("Checking for correct for jurisdiction...")
        jurisdiction_validator = DTreeJurisdictionValidator(
            jurisdiction, **self.kwargs
        )
        return await _validator_check_for_doc(
            validator=jurisdiction_validator,
            doc=doc,
            score_thresh=self.score_thresh,
        )


class JurisdictionWebsiteValidator:
    """COMPASS Ordinance Jurisdiction Website validator"""

    WEB_PAGE_CHECK_SYSTEM_MESSAGE = (
        "You are an expert data analyst that examines website text to "
        "determine if the website is the main website for a given "
        "jurisdiction. Only ever answer based on the information from the "
        "website itself."
    )
    """System message for main jurisdiction website validation calls"""

    def __init__(
        self, browser_semaphore=None, file_loader_kwargs=None, **kwargs
    ):
        """

        Parameters
        ----------
        browser_semaphore : :class:`asyncio.Semaphore`, optional
            Semaphore instance that can be used to limit the number of
            playwright browsers open concurrently. If ``None``, no
            limits are applied. By default, ``None``.
        file_loader_kwargs : dict, optional
            Dictionary of keyword arguments pairs to initialize
            :class:`elm.web.file_loader.AsyncWebFileLoader`.
            By default, ``None``.
        **kwargs
            Additional keyword arguments to pass to the
            :class:`~compass.llm.calling.BaseLLMCaller` instance.

        """
        self.browser_semaphore = browser_semaphore
        self.file_loader_kwargs = file_loader_kwargs or {}
        self.kwargs = kwargs

    async def check(self, url, jurisdiction):
        """Check if the website is the main website for a jurisdiction

        Parameters
        ----------
        url : str
            URL of the website to validate.

        Returns
        -------
        bool
            ``True`` if the website is the main website for the given
            jurisdiction; ``False`` otherwise.
        """

        url_validator = DTreeURLJurisdictionValidator(
            jurisdiction, **self.kwargs
        )

        url_is_correct_jurisdiction = await url_validator.check(url)

        if url_is_correct_jurisdiction:
            return True

        fl = AsyncWebFileLoader(
            browser_semaphore=self.browser_semaphore,
            **self.file_loader_kwargs,
        )
        try:
            doc = await fl.fetch(url)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            msg = "Encountered error of type %r while trying to validate %s"
            err_type = type(e)
            logger.exception(msg, err_type, url)
            return False

        if doc.empty:
            return False

        prompt = (
            "Based on the website text below, is it reasonable to conclude "
            f"that this webpage is the **main** {jurisdiction.type} website "
            f"for {jurisdiction.full_name_the_prefixed}? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain your answer."
            f'\n\n"""\n{doc.text}\n"""'
        )

        local_chat_llm_caller = LLMCaller(**self.kwargs)
        out = await local_chat_llm_caller.call(
            sys_msg=self.WEB_PAGE_CHECK_SYSTEM_MESSAGE,
            content=prompt,
            usage_sub_label=(
                LLMUsageCategory.JURISDICTION_MAIN_WEBSITE_VALIDATION
            ),
        )

        return out.casefold().startswith("yes")


async def _validator_check_for_doc(validator, doc, score_thresh=0.9, **kwargs):
    """Apply a validator check to a doc's raw pages"""
    outer_task_name = asyncio.current_task().get_name()
    validation_checks = [
        asyncio.create_task(
            validator.check(text, **kwargs), name=outer_task_name
        )
        for text in doc.raw_pages
    ]
    out = await asyncio.gather(*validation_checks)
    score = _weighted_vote(out, doc)
    doc.attrs[validator.META_SCORE_KEY] = score
    logger.debug(
        "%s is %.2f for doc from source %s (Pass: %s; threshold: %.2f)",
        validator.META_SCORE_KEY,
        score,
        doc.attrs.get("source", "Unknown"),
        score >= score_thresh,
        score_thresh,
    )
    return score >= score_thresh


def _weighted_vote(out, doc):
    """Compute weighted average of responses based on text length"""
    if not doc.raw_pages:
        return 0

    total = weights = 0
    for verdict, text in zip(out, doc.raw_pages, strict=True):
        if verdict is None:
            continue
        weight = len(text)
        logger.debug("Weight=%d, Verdict=%d", weight, int(verdict))
        weights += weight
        total += verdict * weight

    weights = max(weights, 1)
    return total / weights
