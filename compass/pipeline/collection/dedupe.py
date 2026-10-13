"""Document deduplication for collected artifacts"""

import logging
from collections import UserDict
from dataclasses import dataclass

from elm.web.document import BaseDocument

from compass.utilities.enums import COMPASSDocumentCollectionStep

logger = logging.getLogger(__name__)


@dataclass
class _DocInfo:
    """Information about a collected document"""

    doc: BaseDocument
    from_steps: list[str]

    def add_step(self, step_name: str | None):
        """Add a collection step to the provenance of this document"""
        if step_name and step_name not in self.from_steps:
            self.from_steps.append(step_name)

    @classmethod
    def from_doc(cls, doc: BaseDocument):
        """Create a new _DocInfo from a document"""
        return cls(doc=doc, from_steps=list(doc.attrs.get("from_steps", [])))


class DocumentDeDuplicator(UserDict):
    """Domain Service for deduplicating collected documents"""

    def add_docs(self, docs, *, step_name=None):
        """Add documents to the collection mapping

        Parameters
        ----------
        docs : list
            Collected document objects to add to the internal
            de-duplicated mapping.
        step_name : str, optional
            Identifier for the collection step that produced the
            documents. If not provided, "from_steps" will not be updated
            for the added documents. By default, ``None``.
        """
        if not docs:
            if step_name:
                logger.debug("No docs found to add for step %r", step_name)
            return

        logger.debug("Adding %d doc(s) to collection", len(docs))
        for doc in docs:
            key = _collection_doc_key(doc.attrs)
            doc_info = self.data.setdefault(key, _DocInfo.from_doc(doc))
            _update_to_best_metadata(doc_info, doc, step_name)
            doc_info.add_step(step_name)

    def info(self, doc):
        """Get the info for a given document

        Parameters
        ----------
        doc : BaseDocument
            The document for which to retrieve the deduplication info.

        Returns
        -------
        object
            The deduplication info for the given document.
        """
        return self.data.get(
            _collection_doc_key(doc.attrs), _DocInfo(doc=doc, from_steps=[])
        )


def _collection_doc_key(doc_info):
    """Build the deduplication key for a collected document"""
    try:
        return str(doc_info["checksum"])
    except KeyError:
        return str(
            doc_info.get("checksum")
            or doc_info.get("source_fp")
            or doc_info.get("source")
            or doc_info.get("cache_fn")
            or id(doc_info)
        )


def _update_to_best_metadata(doc_info, doc, step_name):
    """Update the document info to reflect the best metadata"""
    if doc_info.doc is doc:
        return

    _merge_search_engines(doc_info, doc)
    _set_best_collection_rank(doc_info, doc, step_name)


def _merge_search_engines(doc_info, doc):
    """Merge in the search engines from the given document"""
    if "search_engines" not in doc.attrs:
        return

    doc_info.doc.attrs["search_engines"] = list(
        dict.fromkeys(
            doc_info.doc.attrs.get("search_engines", [])
            + doc.attrs["search_engines"]
        )
    )


def _set_best_collection_rank(doc_info, doc, step_name):
    """Set the best collection rank for the document"""
    rank = doc.attrs.get("collection_step_rank")
    existing_rank = doc_info.doc.attrs.get("collection_step_rank")
    if (
        step_name == COMPASSDocumentCollectionStep.SEARCH_ENGINE
        and rank is not None
        and (
            COMPASSDocumentCollectionStep.SEARCH_ENGINE
            not in doc_info.from_steps
            or existing_rank is None
            or rank < existing_rank
        )
    ):
        doc_info.doc.attrs["collection_step_rank"] = rank
