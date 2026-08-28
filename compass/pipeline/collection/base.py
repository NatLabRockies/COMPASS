"""Collection workflow for the COMPASS pipeline"""

import logging
from functools import cached_property

from elm.web.document import BaseDocument

from compass.pipeline.collection.dedupe import DocumentDeDuplicator
from compass.pipeline.collection.steps import (
    CompassWebsiteCrawlStep,
    ElmWebsiteCrawlStep,
    KnownLocalDocumentsStep,
    KnownUrlDocumentsStep,
    SearchEngineDocumentsStep,
)

logger = logging.getLogger(__name__)


class _PersistedDocument(BaseDocument):
    """Document subclass used to hold collection artifacts"""

    WRITE_KWARGS = None
    FILE_EXTENSION = None

    def __init__(self, attrs):
        super().__init__(pages=[], attrs=attrs)

    def _raw_pages(self):
        """Get raw pages from document"""

    def _cleaned_text(self):
        """Compute cleaned text from document"""


class DocumentCollection:
    """Workflow object that applies a fixed pipeline of steps"""

    def __init__(self, workflow):
        """

        Parameters
        ----------
        workflow : compass.pipeline.jurisdiction.SingleJurisdictionRun
            The workflow for the jurisdiction being processed, which may
            or may not have website search enabled. The workflow is
            passed to each collection step, which may use it to access
            jurisdiction information and other relevant data, and to
            determine whether website search is enabled.
        """
        self.workflow = workflow
        self.de_duplicator = DocumentDeDuplicator()
        self._collection_info = {}
        self._completed_steps = set()

    @cached_property
    def steps(self):
        """Collection steps in the order they should be executed"""
        steps = []

        if self.workflow.known_local_docs:
            steps.append(KnownLocalDocumentsStep())
        else:
            logger.debug(
                "%r processing has no known local docs configured",
                self.workflow.jurisdiction.full_name,
            )

        if self.workflow.known_doc_urls:
            steps.append(KnownUrlDocumentsStep())
        else:
            logger.debug(
                "%r processing has no known URLs configured",
                self.workflow.jurisdiction.full_name,
            )

        if self.workflow.perform_se_search:
            steps.append(SearchEngineDocumentsStep())
        else:
            logger.debug(
                "%r processing doesn't have SE search enabled",
                self.workflow.jurisdiction.full_name,
            )

        if self.workflow.perform_website_search:
            steps.extend([ElmWebsiteCrawlStep(), CompassWebsiteCrawlStep()])
        else:
            logger.debug(
                "%r processing doesn't have website search enabled",
                self.workflow.jurisdiction.full_name,
            )

        return steps

    async def execute(self, *, eager_extract=False):
        """Run the fixed collection sequence

        The document collection has a well-defined order:

            1. Process any/all known local documents
            2. Process any/all known document URLs
            3. Search engine-based search for ordinance documents
            4. Jurisdiction website crawl-based search for ordinance
               documents

        Users can disable any of these steps via the workflow
        configuration.

        Parameters
        ----------
        eager_extract : bool, optional
            Option to apply extraction as soon as any documents are
            found. If the extraction returns any structured data,
            subsequent steps are skipped for that jurisdiction.
            By default, ``False``.

        Returns
        -------
        dict or None
            If ``eager_extract`` is ``False``, a dictionary containing
            collection information and metadata. If ``eager_extract`` is
            ``True``, the result of the extraction workflow if any
            structured data was extracted, or ``None`` if no structured
            data was extracted from any of the collected documents.
        """
        await self._load_persisted_docs()
        for step in self._unfinished_steps():
            docs = await self._run_collection_step(step)
            if eager_extract:
                context = (
                    await self.workflow.extraction_workflow.extract_from_docs(
                        docs
                    )
                )
                if context is not None:
                    return self._context_with_documented_steps(context)
            else:
                self._collection_info = (
                    await self.workflow.write_collection_shard_no_fail(
                        self.de_duplicator, self._completed_steps
                    )
                )

        if eager_extract:
            return None

        self._log_execute_results()
        return self._collection_info

    async def _load_persisted_docs(self):
        """Get any previously persisted documents and completed steps"""
        self._collection_info = (
            await self.workflow.load_existing_collection_shard()
        ) or {}

        docs = [
            _PersistedDocument(doc_info)
            for doc_info in self._collection_info.get("documents", [])
        ]
        self.de_duplicator.add_docs(docs)

        self._completed_steps |= set(
            self._collection_info.get("completed_step_document_counts", {})
        )

    def _unfinished_steps(self):
        """Yield unfinished collection steps"""
        for step in self.steps:
            if step.STEP_NAME in self._completed_steps:
                logger.info(
                    "Skipping completed collection step %s for %s",
                    step.STEP_NAME,
                    self.workflow.jurisdiction.full_name,
                )
                continue
            yield step

    async def _run_collection_step(self, step):
        """Run collection step and record results"""
        docs = await step.collect(self.workflow)
        self.de_duplicator.add_docs(docs, step_name=str(step.STEP_NAME))
        self._completed_steps.add(step.STEP_NAME)
        return docs

    def _context_with_documented_steps(self, context):
        """Attach collection steps to each document in the context"""
        for doc in context.data_docs:
            doc.attrs["from_steps"] = list(
                self.de_duplicator.info(doc).from_steps
            )
        return context

    def _log_execute_results(self):
        """Log the results of the collection execution"""
        if self.de_duplicator:
            logger.debug(
                "Collected the following documents for %s:\n\n%s",
                self.workflow.jurisdiction.full_name,
                "\n\n".join(
                    [f"{info.doc!r}" for info in self.de_duplicator.values()]
                ),
            )
        else:
            logger.debug(
                "No documents were collected for %s",
                self.workflow.jurisdiction.full_name,
            )
