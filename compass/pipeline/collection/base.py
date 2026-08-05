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
        completed_steps = await self._load_persisted_docs()

        collection_info = None
        for step in self.steps:
            if step in completed_steps:
                logger.info(
                    "Skipping completed collection step %r for %s",
                    step.STEP_NAME,
                    self.workflow.jurisdiction.full_name,
                )
                continue

            docs = await step.collect(self.workflow)
            self.de_duplicator.add_docs(docs, step_name=str(step.STEP_NAME))
            completed_steps.add(step.STEP_NAME)
            if eager_extract:
                context = (
                    await self.workflow.extraction_workflow.extract_from_docs(
                        docs
                    )
                )
                if context is not None:
                    return context
            else:
                collection_info = (
                    await self.workflow.write_collection_shard_no_fail(
                        self.de_duplicator, completed_steps
                    )
                )

        return collection_info

    async def _load_persisted_docs(self):
        """Get any previously persisted documents and completed steps"""
        existing_collection_info = (
            await self.workflow.load_existing_collection_shard()
        ) or {}
        docs = [
            _PersistedDocument(doc_info)
            for doc_info in existing_collection_info.get("documents", [])
        ]
        self.de_duplicator.add_docs(docs)
        return set(
            existing_collection_info.get("completed_step_document_counts", {})
        )
