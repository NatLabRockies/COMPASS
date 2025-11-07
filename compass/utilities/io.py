"""COMPASS I/O utilities"""

import pprint
import logging

from elm.web.file_loader import AsyncLocalFileLoader


logger = logging.getLogger(__name__)


async def load_local_docs(fps, **kwargs):
    """Load a document for each input filepath

    Parameters
    ----------
    fps : iterable of path-like
        Iterable of paths representing documents to load.
    kwargs
        Keyword-argument pairs to initialize
        :class:`elm.web.file_loader.AsyncLocalFileLoader`.

    Returns
    -------
    list
        List of non-empty document instances containing information from
        the local documents. If a file could not be loaded (i.e.
        document instance is empty), it will not be included in the
        output list.
    """
    logger.trace("Loading docs for the following paths:\n%r", fps)
    logger.trace(
        "kwargs for AsyncLocalFileLoader:\n%s",
        pprint.PrettyPrinter().pformat(kwargs),
    )
    file_loader = AsyncLocalFileLoader(**kwargs)
    docs = await file_loader.fetch_all(*fps)

    page_lens = {
        doc.attrs.get("source_fp", "Unknown"): len(doc.pages) for doc in docs
    }
    logger.debug(
        "Loaded the following number of pages for docs:\n%s",
        pprint.PrettyPrinter().pformat(page_lens),
    )
    return [doc for doc in docs if not doc.empty]
