"""COMPASS I/O utilities"""

import pprint
import logging

from elm.web.file_loader import AsyncLocalFileLoader


logger = logging.getLogger(__name__)


async def load_local_docs(fps, **kwargs):
    """Load local documents into `elm` document instances

    Parameters
    ----------
    fps : Iterable
        Iterable of paths referencing local files to load.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`elm.web.file_loader.AsyncLocalFileLoader` for
        configuration such as ``loader``, caching, or parsing options.

    Returns
    -------
    list of elm.web.document.BaseDocument
        Non-empty loaded documents corresponding to the supplied
        filepaths. Empty results (e.g., unreadable files) are filtered
        out of the returned list.

    Raises
    ------
    elm.exceptions.ELMError
        Propagated when the underlying loader fails to read one of the
        provided files and is configured to raise on errors.

    Notes
    -----
    Detailed debug information about loaded page counts is emitted via
    the ``compass.utilities.io`` logger at ``TRACE`` level to assist
    with troubleshooting ingestion runs.
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
