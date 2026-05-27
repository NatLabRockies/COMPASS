"""Document deduplication for collected artifacts"""

import logging


logger = logging.getLogger(__name__)


class DocumentDeDuplicator:
    """Domain Service for deduplicating collected documents"""

    def __init__(self):
        self._docs = {}

    def add_docs(
        self,
        docs,
        *,
        step_name,
        needs_jurisdiction_verification,
        jurisdiction_name,
    ):
        """Add documents to the collection mapping

        Parameters
        ----------
        docs : list
            Collected document objects to add to the internal
            de-duplicated mapping.
        step_name : str
            Identifier for the collection step that produced the
            documents.
        needs_jurisdiction_verification : bool
            Whether the added documents should be flagged for later
            jurisdiction verification.
        jurisdiction_name : str
            Full jurisdiction name to attach to documents that do not
            already include one.
        """
        if not docs:
            logger.debug("No docs found to add for step %r", step_name)
            return

        logger.debug("Adding %d doc(s) to collection", len(docs))
        for doc in docs:
            doc.attrs.setdefault("jurisdiction_name", jurisdiction_name)
            doc.attrs["check_correct_jurisdiction"] = (
                needs_jurisdiction_verification
            )
            try:
                key = _collection_doc_key(doc)
            except KeyError:
                key = _collection_doc_key(doc, use_fallback=True)

            if key not in self._docs:
                self._docs[key] = {"doc": doc, "from_steps": []}

            self._docs[key]["from_steps"].append(step_name)

    @property
    def values(self):
        """Deduplicated collected docs"""
        return self._docs.values()

    def __bool__(self):
        return bool(self._docs)


def _collection_doc_key(doc, use_fallback=False):
    """Build the deduplication key for a collected document"""
    if use_fallback:
        return str(
            doc.attrs.get("checksum")
            or doc.attrs.get("source_fp")
            or doc.attrs.get("source")
            or doc.attrs.get("cache_fn")
            or id(doc)
        )
    return str(doc.attrs["checksum"])
