"""Extraction context for multi-document ordinance extraction"""

from textwrap import shorten
from collections.abc import Iterable

import pandas as pd

from compass.services.threaded import FileMover
from compass.exceptions import COMPASSTypeError


class ExtractionContext:
    """Context for extraction operations supporting multiple documents

    This class provides a Document-compatible interface for extraction
    workflows that may involve one or more source documents. It tracks
    chunk-level provenance to identify which document each text chunk
    originated from, while maintaining compatibility with existing
    extraction functions that expect Document-like objects
    """

    def __init__(self, documents=None, attrs=None):
        """

        Parameters
        ----------
        documents : sequence of BaseDocument, optional
            One or more source documents contributing to this context.
            For single-document workflows (solar, wind), pass a list
            with one document. For multi-document workflows (water
            rights), pass all contributing documents
        attrs : dict, optional
            Context-level attributes for extraction metadata
            (jurisdiction, tech type, etc.). By default, ``None``
        """
        self.attrs = attrs or {}
        self._documents = _as_list(documents)
        self._data_docs = []

    @property
    def text(self):
        """str: Concatenated text from all documents"""
        return self.multi_doc_context()

    @property
    def pages(self):
        """list: Concatenated pages from all documents"""
        return [page for doc in self.documents for page in doc.pages]

    @property
    def num_documents(self):
        """int: Number of source documents in this context"""
        return len(self.documents)

    @property
    def documents(self):
        """list: List of documents that might contain relevant info"""
        return self._documents

    @documents.setter
    def documents(self, other):
        self._documents = _as_list(other)

    @property
    def data_docs(self):
        """list: List of documents that contributed to extraction"""
        return self._data_docs

    @data_docs.setter
    def data_docs(self, other):
        if not isinstance(other, list):
            msg = "data_docs must be set to a *list* of documents"
            raise COMPASSTypeError(msg)

        self._data_docs = other

    def __str__(self):
        header = (
            f"{self.__class__.__name__} with {self.num_documents:,} document"
        )
        if self.num_documents != 1:
            header = f"{header}s"

        if self.num_documents > 0:
            docs = "\n\t- ".join(
                [
                    d.attrs.get("source", "Unknown source")
                    for d in self.documents
                ]
            )
            header = f"{header}:\n\t- {docs}"

        data_docs = _data_docs_repr(self.data_docs)
        attrs = _attrs_repr(self.attrs)
        return f"{header}\n{data_docs}\n{attrs}"

    def __len__(self):
        return self.num_documents

    def __getitem__(self, index):
        return self.documents[index]

    def __iter__(self):
        return iter(self.documents)

    def __bool__(self):
        return bool(self.documents)

    async def mark_doc_as_data_source(self, doc, out_fn_stem=None):
        """Mark a document as a data source for extraction

        Parameters
        ----------
        doc : BaseDocument
            Document to add as a data source
        out_fn_stem : str, optional
            Optional output filename stem for this document. If
            provided, the document file will be moved from the
            temporary directory to the output directory with this
            filename stem and appropriate file suffix.
            By default, ``None``.
        """
        self._data_docs.append(doc)
        if out_fn_stem is not None:
            await _move_file_to_out_dir(doc, out_fn_stem)

    def multi_doc_context(self, attr_text_key=None):
        """Get concatenated text representation of documents

        This method creates a concatenated text representation of the
        documents in this context, optionally pulling the text from the
        documents' `attr_text_key`.

        Parameters
        ----------
        attr_text_key : str, optional
            The key under which the concatenated text will be stored in
            the context attributes.

        Returns
        -------
        str
            Concatenated text representation of the documents in this
            context.
        """
        if not self.documents:
            return ""

        serialized = "\n\n".join(
            (
                f"# SOURCE INDEX #: {ind}\n"
                f"# CONTENT #:\n{_text_from_doc(doc, attr_text_key)}"
            )
            for ind, doc in enumerate(self.documents)
        )
        return f"## MULTI-DOCUMENT CONTEXT ##\n\n{serialized}"


async def _move_file_to_out_dir(doc, out_fn):
    """Move PDF or HTML text file to output directory"""
    out_fp = await FileMover.call(doc, out_fn)
    doc.attrs["out_fp"] = out_fp
    return doc


def _as_list(documents):
    """Convert input to list"""
    if documents is None:
        return []
    if not isinstance(documents, Iterable):
        return [documents]
    return list(documents)


def _data_docs_repr(data_docs):
    """String representation of data source documents"""
    if not data_docs:
        return "Registered Data Source Documents: None"

    data_docs = "\n\t- ".join(
        [d.attrs.get("source", "Unknown source") for d in data_docs]
    )

    return f"Registered Data Source Documents:\n\t- {data_docs}"


def _attrs_repr(attrs):
    """String representation of context attributes"""
    if not attrs:
        return "Attrs: None"

    attrs = {
        k: (
            f"DataFrame with {len(v):,} rows"
            if isinstance(v, pd.DataFrame)
            else v
        )
        for k, v in attrs.items()
    }

    indent = max(len(k) for k in attrs) + 2
    width = max(10, 80 - (indent + 4))
    to_join = []
    for k, v in attrs.items():
        v_str = str(v)
        if "\n" in v_str:
            v_str = shorten(v_str, width=width)
        to_join.append(f"{k:>{indent}}:\t{v_str}")

    attrs = "\n".join(to_join)
    return f"Attrs:\n{attrs}"


def _text_from_doc(doc, key):
    """Get text from key or full doc"""
    return doc.text if key is None else doc.attrs[key]
