"""Extraction workflow for prepared documents"""

import logging

from compass.extraction.context import ExtractionContext
from compass.services.threaded import OrdDBFileWriter
from compass.pb import COMPASS_PB


logger = logging.getLogger(__name__)


class DocumentExtraction:
    """Workflow object that follows a fixed extraction pipeline"""

    def __init__(self, workflow):
        self.workflow = workflow

    async def extract_from_docs(
        self, docs, *, needs_jurisdiction_verification=True
    ):
        """Filter and extract data from a set of docs

        Parameters
        ----------
        docs : iterable
            The documents to filter and extract structured data from.
        needs_jurisdiction_verification : bool, optional
            Flag indicating whether the jurisdiction of the document
            needs to be verified. By default, ``True``.

        Returns
        -------
        compass.extraction.context.ExtractionContext or None
            The context containing extracted structured data and other
            relevant information, or ``None`` if no data was extracted.
        """
        if not docs:
            return None

        extraction_context = ExtractionContext(documents=docs)
        extraction_context = await self.workflow.extractor.filter_docs(
            extraction_context,
            needs_jurisdiction_verification=needs_jurisdiction_verification,
        )
        if not extraction_context:
            return None

        extraction_context.attrs["jurisdiction_website"] = (
            self.workflow.jurisdiction_website
        )

        COMPASS_PB.update_jurisdiction_task(
            self.workflow.jurisdiction.full_name,
            description="Extracting structured data...",
        )
        context = await self.workflow.extractor.parse_docs_for_structured_data(
            extraction_context
        )
        await self._write_out_structured_data(extraction_context)
        logger.debug("Final extraction context:\n%s", context)
        return context

    async def _write_out_structured_data(self, extraction_context):
        """Write structured output for one jurisdiction"""
        if extraction_context.attrs.get("structured_data") is None:
            return

        out_fn = extraction_context.attrs.get("out_data_fn")
        if out_fn is None:
            out_fn = f"{self.workflow.jurisdiction.full_name} Ordinances.csv"

        out_fp = await OrdDBFileWriter.call(extraction_context, out_fn)
        logger.info(
            "Structured data for %s stored here: '%s'",
            self.workflow.jurisdiction.full_name,
            out_fp,
        )
        extraction_context.attrs["ord_db_fp"] = out_fp
