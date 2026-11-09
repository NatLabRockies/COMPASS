"""Extraction workflow for prepared documents"""

import logging

from compass.extraction.context import ExtractionContext
from compass.services.threaded import OrdDBFileWriter
from compass.pb import COMPASS_PB
from compass.utilities.timing import log_operation
from compass.utilities.event_log import (
    document_summary,
    emit_branch,
    emit_event,
    event_phase,
    event_step,
)


logger = logging.getLogger(__name__)


class DocumentExtraction:
    """Workflow object that follows a fixed extraction pipeline"""

    def __init__(self, workflow):
        self.workflow = workflow

    @property
    def max_docs_to_parse(self):
        """int or None: Maximum number of documents to parse"""
        parse_settings = self.workflow.runtime.request.parsing_settings
        return parse_settings.max_num_docs_per_jurisdiction

    async def extract_from_docs(self, docs):
        """Filter and extract data from a set of docs

        Parameters
        ----------
        docs : iterable
            The documents to filter and extract structured data from.

        Returns
        -------
        compass.extraction.context.ExtractionContext or None
            The context containing extracted structured data and other
            relevant information, or ``None`` if no data was extracted.
        """
        if not docs:
            emit_branch("handoff", "no")
            return None

        emit_branch("handoff", "yes")
        extraction_context = ExtractionContext(documents=docs)
        with event_step("filter"), event_phase("filter"):
            emit_event(
                "documents_in",
                documents=[document_summary(doc) for doc in docs],
            )
            async with log_operation(
                logger,
                "document filtering",
                self.workflow.jurisdiction.full_name,
            ):
                extraction_context = await self.workflow.extractor.filter_docs(
                    extraction_context, self.max_docs_to_parse
                )
            with event_step("filter.order"):
                emit_event(
                    "documents_kept",
                    documents=[
                        document_summary(doc)
                        for doc in (
                            extraction_context.documents
                            if extraction_context
                            else []
                        )
                    ],
                )
            emit_branch(
                "filter.any_kept", "yes" if extraction_context else "no"
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
        with event_step("extract"), event_phase("extract"):
            async with log_operation(
                logger,
                "structured extraction",
                self.workflow.jurisdiction.full_name,
            ):
                context = await (
                    self.workflow.extractor.parse_docs_for_structured_data(
                        extraction_context
                    )
                )
            _emit_extracted_rows(extraction_context)
        with event_step("save"), event_phase("save"):
            async with log_operation(
                logger,
                "saving extraction",
                self.workflow.jurisdiction.full_name,
            ):
                await self._write_out_structured_data(extraction_context)
            ord_db_fp = extraction_context.attrs.get("ord_db_fp")
            with event_step("save.write" if ord_db_fp else "save.skip"):
                emit_event("saved", ord_db_fp=str(ord_db_fp))
        logger.debug("Final extraction context:\n%s", context)
        return context

    async def _write_out_structured_data(self, extraction_context):
        """Write structured output for one jurisdiction"""
        if extraction_context.attrs.get("structured_data") is None:
            emit_branch("save.has_data", "no")
            return

        emit_branch("save.has_data", "yes")

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


def _emit_extracted_rows(extraction_context):
    """Record the structured rows extracted for the jurisdiction"""
    data = extraction_context.attrs.get("structured_data")
    if data is None:
        emit_event("extracted_rows", count=0, rows=[])
        return
    rows = data.to_dict(orient="records") if hasattr(data, "to_dict") else []
    emit_event(
        "extracted_rows",
        count=len(rows),
        columns=list(getattr(data, "columns", [])),
        rows=rows,
    )
