"""Document deduplication for collected artifacts"""

import logging
from collections import UserDict


logger = logging.getLogger(__name__)


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
            entry = self.data.setdefault(
                key,
                {
                    "doc": doc,
                    "from_steps": list(doc.attrs.get("from_steps", [])),
                },
            )
            if step_name and step_name not in entry["from_steps"]:
                entry["from_steps"].append(step_name)


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
