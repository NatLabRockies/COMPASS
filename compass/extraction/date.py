"""Ordinance date extraction logic"""

import logging
from datetime import datetime

from compass.utilities.enums import LLMUsageCategory
from compass.utilities.parsing import raw_pages_from_doc


logger = logging.getLogger(__name__)


class DateExtractor:
    """Helper class to extract date info from document"""

    SYSTEM_MESSAGE = (
        "You are a legal scholar that reads ordinance text and extracts "
        "the date the ordinance most recently took legal effect. "
        "Ordinances are often adopted once and then amended or updated "
        "over time. Report the LATEST date on which the ordinance was "
        "adopted, enacted, passed, amended, revised, updated, or became "
        "effective. For example, if the text says it was adopted in one "
        "year but 'updated through' or 'last updated' a later year, "
        "report the later year. "
        "Look for this date in the title, preamble, signature/adoption "
        "block, an 'Ordinance No.' line, amendment history, or a "
        "'last updated' / 'updated through' note. "
        "Do NOT report any of the following, even if they are the most "
        "prominent dates in the text: meeting, agenda, or public hearing "
        "dates; draft dates; or the date the file was uploaded, "
        "downloaded, scanned, or collected from a website. If a date "
        "appears only in a page header or footer and looks like a "
        "file-collection or publication stamp (rather than a stated "
        "adoption or update date), do not report it. "
        "Language describing when the ordinance becomes effective (for "
        "example, that it takes effect upon or after its passage, "
        "adoption, approval, signing, publication, recording, or filing, "
        "or some number of days afterward) is normal in enacted "
        "ordinances. Do not treat such conditional or future-effective "
        "wording as evidence that the document is a draft or was never "
        "enacted; report the date it was passed, adopted, signed, or "
        "otherwise took effect. "
        "A URL or file name for the document may be provided as a hint, "
        "but the document text is the source of truth. Use the hint to "
        "corroborate or complete a date the text partially supports: if "
        "the text clearly identifies a passage/adoption/amendment event "
        "but is missing part of its date (for example the year is blank "
        "or not printed), and a date in the URL or file name is "
        "consistent with the part the text does state, you may use the "
        "hint to fill in the missing piece. Do not use a date from the "
        "URL or file name that conflicts with the date stated in the "
        "text, and do not invent a date the text gives no support for. "
        "Return your answer as a dictionary in JSON format (not "
        "markdown). Your JSON file must include exactly four keys. The "
        "first key is 'explanation', which contains a short summary of "
        "the date information you found, including the exact text the "
        "date is based on. The second key is 'year', which should "
        "contain an integer value for the latest year the ordinance took "
        "effect, or null if that cannot be confidently determined from "
        "the text. The third key is 'month', which should contain an "
        "integer value for the corresponding month, or null. The fourth "
        "key is 'day', which should contain an integer value for the "
        "corresponding day of the month, or null. Only provide a value "
        "if you are confident it represents the latest date this "
        "ordinance took effect; otherwise use null."
    )
    """System message for date extraction LLM calls"""

    def __init__(self, json_llm_caller, text_splitter=None):
        """

        Parameters
        ----------
        json_llm_caller : JSONFromTextLLMCaller
            Instance used for structured validation queries.
        text_splitter : LCTextSplitter, optional
            Optional text splitter (or subclass instance, or any object
            that implements a `split_text` method) to attach to doc
            (used for splitting out pages in an HTML document).
            By default, ``None``.
        """
        self.jlc = json_llm_caller
        self.text_splitter = text_splitter

    async def parse(self, doc):
        """Extract date (year, month, day) from doc

        The full document text is read in a single LLM call. The
        document's ``source`` URL, if any, is passed along as a hint,
        but the document text is the source of truth.

        Parameters
        ----------
        doc : BaseDocument
            Document to parse.

        Returns
        -------
        tuple
            3-tuple containing year, month, day, or ``None`` if any of
            those are not found.
        """
        raw_pages = raw_pages_from_doc(doc, self.text_splitter)
        text = "\n\n".join(page for page in raw_pages if page)
        if not text:
            return None, None, None

        content = "Please extract the enactment date for this ordinance."
        url = doc.attrs.get("source")
        if url:
            content += f"\nThe document was downloaded from this URL: {url}"
        content += f"\n\nOrdinance text:\n{text}"

        response = await self.jlc.call(
            sys_msg=self.SYSTEM_MESSAGE,
            content=content,
            usage_sub_label=LLMUsageCategory.DATE_EXTRACTION,
        )
        if response:
            logger.debug(
                "Date extraction explanation: %s",
                response.get("explanation"),
            )
        date = _parse_date(response)
        logger.debug("Parsed date: %s", date)
        return date


def _parse_date(date_info):
    """Validate and return the (year, month, day) from a response"""
    if not date_info:
        return None, None, None

    year = _validated_element(
        date_info, key="year", min_val=2000, max_val=datetime.now().year
    )
    month = _validated_element(
        date_info, key="month", min_val=1, max_val=12
    )
    day = _validated_element(date_info, key="day", min_val=1, max_val=31)
    return year, month, day


def _validated_element(date_info, key, min_val, max_val):
    """Return a single date element if it falls within the valid range

    Acts as a cheap safety net against an out-of-range or malformed
    value in the model response. Coerces the value to an int (so a
    numeric string like "2020" or a float like 2020.0 is accepted) and
    returns ``None`` if the value is missing, non-numeric, or outside
    ``[min_val, max_val]``.
    """
    value = date_info.get(key)
    logger.debug("key=%r, value=%r", key, value)
    if value is None:
        return None
    try:
        value = int(float(value))
    except (TypeError, ValueError):
        return None
    return value if min_val <= value <= max_val else None
