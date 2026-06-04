"""Ordinance date extraction logic"""

import logging
from datetime import datetime

from compass.utilities.enums import LLMUsageCategory
from compass.utilities.parsing import raw_pages_from_doc


logger = logging.getLogger(__name__)

# Cap the text sent in a single date-extraction call. Enactment dates
# live in the preamble (front) and signature/adoption block (back), so
# keep the first and last pages and drop the middle for long documents.
_MAX_HEAD_PAGES = 20
_MAX_TAIL_PAGES = 10


class DateExtractor:
    """Helper class to extract date info from document"""

    SYSTEM_MESSAGE = (
        "You are a legal scholar that reads ordinance text and extracts "
        "a single date. The date you report is the latest of two things: "
        "(1) the date the ordinance was most recently enacted (adopted, "
        "passed, amended, revised, or made effective), or (2) the date "
        "the official document containing the enacted ordinance was most "
        "recently updated. Report whichever of these is later. "
        "For (1), look in the title, preamble, signature/adoption block, "
        "an 'Ordinance No.' line, or the amendment history; if the text "
        "says it was adopted in one year but 'updated through' a later "
        "year, use the later year. "
        "For (2), an official document-update date is a date the "
        "publisher of the codified ordinance gives for the document "
        "itself, such as a 'Last Updated', 'Last Revised', 'Current "
        "through', or codification date printed on the document. Treat "
        "this as a valid answer when it is later than any enactment date "
        "in the text. "
        "Do NOT report any of the following, even if they are the most "
        "prominent dates in the text: meeting, agenda, or public hearing "
        "dates; draft dates; or incidental file-handling dates such as "
        "when this particular file was uploaded, downloaded, scanned, or "
        "collected from a website (for example a CMS upload path like "
        "'/uploads/2017/09/'). The distinction for (2) is whether the "
        "date describes when the official ordinance document was revised "
        "(report it) versus when this copy of the file happened to be "
        "produced or fetched (ignore it). "
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
        "contain an integer value for the latest year the ordinance was "
        "enacted or the official document was updated, or null if that "
        "cannot be confidently determined from the text. The third key "
        "is 'month', which should contain an integer value for the "
        "corresponding month, or null. The fourth key is 'day', which "
        "should contain an integer value for the corresponding day of "
        "the month, or null. Only provide a value if you are confident "
        "it represents that latest date; otherwise use null."
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
            3-tuple of (year, month, day). Any element that cannot be
            determined is ``None``.
        """
        raw_pages = [
            page
            for page in raw_pages_from_doc(doc, self.text_splitter)
            if page
        ]
        raw_pages = _trim_pages(raw_pages)
        text = "\n\n".join(raw_pages)
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


def _trim_pages(pages):
    """Keep the head and tail pages, dropping the middle if too long"""
    if len(pages) <= _MAX_HEAD_PAGES + _MAX_TAIL_PAGES:
        return pages
    return pages[:_MAX_HEAD_PAGES] + pages[-_MAX_TAIL_PAGES:]


def _parse_date(date_info):
    """Validate and return the (year, month, day) from a response"""
    if not date_info:
        return None, None, None

    year = _validated_element(
        date_info, key="year", min_val=1950, max_val=datetime.now().year + 1
    )
    month = _validated_element(date_info, key="month", min_val=1, max_val=12)
    day = _validated_element(date_info, key="day", min_val=1, max_val=31)
    return year, month, day


def _validated_element(date_info, key, min_val, max_val):
    """Return a single date element if it falls within the valid range

    Acts as a cheap safety net against an out-of-range or malformed
    value in the model response.
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
