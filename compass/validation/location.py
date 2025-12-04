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
    """Validate whether a URL appears to target a jurisdiction"""

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
        jurisdiction : compass.utilities.location.Jurisdiction
            Jurisdiction descriptor with the target location attributes.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`~compass.llm.calling.BaseLLMCaller` for model
            selection, temperature, or tracing control.

        Notes
        -----
        The validator stores the input jurisdiction for subsequent URL
        checks; it does not perform any validation work during
        instantiation.
        """
        super().__init__(**kwargs)
        self.jurisdiction = jurisdiction

    async def check(self, url):
        """Determine whether the supplied URL targets the jurisdiction

        Parameters
        ----------
        url : str
            URL string to evaluate. Empty values short-circuit to
            ``False``.

        Returns
        -------
        bool
            ``True`` when the decision-tree evaluation finds all
            jurisdiction criteria satisfied, ``False`` otherwise.

        Raises
        ------
        compass.exceptions.COMPASSError
            Propagated if underlying LLM interactions fail while the
            caller has configured
            :class:`~compass.llm.calling.BaseLLMCaller` to raise.

        Notes
        -----
        The method delegates to an internal asynchronous decision tree
        backed by :class:`~compass.llm.calling.ChatLLMCaller`. The
        validator aggregates structured responses and only approves when
        each required attribute matches the target jurisdiction.
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
    """Validate ordinance text against a target jurisdiction"""

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
        jurisdiction : compass.utilities.location.Jurisdiction
            Jurisdiction descriptor identifying expected applicability.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`~compass.llm.calling.BaseLLMCaller` for configuring
            LLM temperature, timeout, or similar options.
        """
        super().__init__(**kwargs)
        self.jurisdiction = jurisdiction

    async def check(self, content):
        """Determine whether ordinance text matches the jurisdiction

        The decision tree checks jurisdiction type, state, and
        subdivision alignment.

        Parameters
        ----------
        content : str
            Plain-text ordinance content extracted from a document.

        Returns
        -------
        bool
            ``True`` when the decision tree concludes the ordinance is
            scoped to the configured jurisdiction, ``False`` otherwise.

        Raises
        ------
        compass.exceptions.COMPASSError
            Raised if the underlying LLM caller propagates an execution
            failure.

        Notes
        -----
        Empty content returns ``False`` without invoking the LLM.
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
    """Coordinate URL and text jurisdiction validation for documents

    Notes
    -----
    The validator stores the score threshold, optional text splitter,
    and keyword arguments so they can be reused across many documents
    without reconfiguration.
    """

    def __init__(self, score_thresh=0.8, text_splitter=None, **kwargs):
        """

        Parameters
        ----------
        score_thresh : float, optional
            Threshold applied to the weighted page vote. Documents at or
            above the threshold are considered jurisdiction matches.
            Default is ``0.8``.
        text_splitter : LCTextSplitter, optional
            Optional splitter attached to documents lacking a
            ``text_splitter`` attribute so validators can iterate page
            content consistently. Default is ``None``.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`~compass.llm.calling.BaseLLMCaller` and reused when
            instantiating subordinate validators.
        """
        self.score_thresh = score_thresh
        self.text_splitter = text_splitter
        self.kwargs = kwargs

    async def check(self, doc, jurisdiction):
        """Assess whether a document applies to the jurisdiction

        Parameters
        ----------
        doc : elm.web.document.BaseDocument
            Document to evaluate. The validator expects
            ``doc.raw_pages`` and, when available, a
            ``doc.attrs['source']`` URL for supplemental URL validation.
        jurisdiction : compass.utilities.location.Jurisdiction
            Target jurisdiction descriptor capturing the required
            location attributes.

        Returns
        -------
        bool
            ``True`` when either the URL or document text validation
            confirms jurisdiction alignment, ``False`` otherwise.

        Raises
        ------
        compass.exceptions.COMPASSError
            Propagated if subordinate validators encounter LLM caller
            errors.

        Notes
        -----
        The method temporarily overrides ``doc.text_splitter`` when a
        custom splitter is provided, ensuring the original splitter is
        restored after validation completes.

        Examples
        --------
        >>> validator = JurisdictionValidator()
        >>> await validator.check(document, jurisdiction)
        True
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
    """Validate whether a website is the primary jurisdiction portal

    Notes
    -----
    The validator stores the initialization arguments so they can be
    reused across many documents without reconfiguration.
    """

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
        browser_semaphore : asyncio.Semaphore, optional
            Semaphore constraining concurrent Playwright usage.
            ``None`` applies no concurrency limit. Default is ``None``.
        file_loader_kwargs : dict, optional
            Keyword arguments passed to
            :class:`elm.web.file_loader.AsyncWebFileLoader`. Default is
            ``None``.
        **kwargs
            Additional keyword arguments cached for downstream LLM
            calls triggered during validation.
        """
        self.browser_semaphore = browser_semaphore
        self.file_loader_kwargs = file_loader_kwargs or {}
        self.kwargs = kwargs

    async def check(self, url, jurisdiction):
        """Determine whether a website serves as a jurisdiction's portal

        The validator first performs an inexpensive URL classification
        before downloading page content. Only when the URL fails the
        initial check does it fetch and inspect the page text using a
        generic LLM caller.

        Parameters
        ----------
        url : str
            URL to inspect. Empty values return ``False`` immediately.
        jurisdiction : compass.utilities.location.Jurisdiction
            Target jurisdiction descriptor used to frame the validation
            prompts.

        Returns
        -------
        bool
            ``True`` when either the URL quick check or the full page
            evaluation indicates the site is the official main website
            for the jurisdiction.

        Raises
        ------
        compass.exceptions.COMPASSError
            Propagated from :class:`~compass.llm.calling.BaseLLMCaller`
            if configured to raise on LLM failures.

        Examples
        --------
        >>> validator = JurisdictionWebsiteValidator()
        >>> await validator.check("https://county.gov", jurisdiction)
        True
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
