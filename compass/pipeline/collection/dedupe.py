"""Document deduplication for collected artifacts"""

import logging


logger = logging.getLogger(__name__)


class DocumentDeDuplicator:
    """Domain Service for deduplicating collected documents"""

    def __init__(self):
        self._docs = {}

    def add_docs(self, docs, *, step_name, jurisdiction_name):
        """Add documents to the collection mapping

        Parameters
        ----------
        docs : list
            Collected document objects to add to the internal
            de-duplicated mapping.
        step_name : str
            Identifier for the collection step that produced the
            documents.
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
            key = _collection_doc_key(doc.attrs)
            entry = self._docs.setdefault(key, {"doc": doc, "from_steps": []})
            if step_name not in entry["from_steps"]:
                entry["from_steps"].append(step_name)

    @property
    def values(self):
        """Deduplicated collected docs"""
        return self._docs.values()

    def __bool__(self):
        return bool(self._docs)


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
