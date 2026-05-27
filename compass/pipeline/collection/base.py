"""Collection workflow for the COMPASS pipeline"""

from compass.pipeline.collection.dedupe import DocumentDeDuplicator
from compass.pipeline.collection.persistence import persist_documents
from compass.pipeline.collection.steps import (
    CompassWebsiteCrawlStep,
    ElmWebsiteCrawlStep,
    KnownLocalDocumentsStep,
    KnownUrlDocumentsStep,
    SearchEngineDocumentsStep,
)


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
        self.steps = [
            KnownLocalDocumentsStep(),
            KnownUrlDocumentsStep(),
            SearchEngineDocumentsStep(),
            ElmWebsiteCrawlStep(),
            CompassWebsiteCrawlStep(),
        ]

    async def execute(self, *, eager_extract=False, relative_to=None):
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
        relative_to : path-like, optional
            Optional directory that should be the root of all relative
            paths. By default, ``None``.

        Returns
        -------
        dict or None
            If ``eager_extract`` is ``False``, a dictionary containing
            collection information and metadata. If ``eager_extract`` is
            ``True``, the result of the extraction workflow if any
            structured data was extracted, or ``None`` if no structured
            data was extracted from any of the collected documents.
        """
        for step in self.steps:
            docs = await step.collect(self.workflow)
            self.de_duplicator.add_docs(
                docs,
                step_name=str(step.STEP_NAME),
                needs_jurisdiction_verification=(
                    step.NEEDS_JURISDICTION_VERIFICATION
                ),
                jurisdiction_name=self.workflow.jurisdiction.full_name,
            )
            if eager_extract:
                context = (
                    await self.workflow.extraction_workflow.extract_from_docs(
                        docs,
                        needs_jurisdiction_verification=(
                            step.NEEDS_JURISDICTION_VERIFICATION
                        ),
                    )
                )
                if context is not None:
                    return context

        if eager_extract:
            return None

        collection_info = await persist_documents(
            self.workflow.jurisdiction,
            self.de_duplicator,
            relative_to=relative_to,
        )
        collection_info["jurisdiction_website"] = (
            self.workflow.jurisdiction_website
        )
        return collection_info
