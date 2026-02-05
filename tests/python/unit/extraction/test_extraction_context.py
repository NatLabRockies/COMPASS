"""COMPASS extraction context tests"""

from pathlib import Path

import pandas as pd
import pytest
from elm.web.document import PDFDocument, HTMLDocument

from compass.extraction.context import (
    ExtractionContext,
    _as_list,
    _attrs_repr,
    _data_docs_repr,
    _move_file_to_out_dir,
)
from compass.exceptions import COMPASSTypeError
from compass.services.threaded import FileMover


def test_extraction_context_iter_empty():
    """Test empty ExtractionContext iteration"""
    for __ in ExtractionContext():
        msg = "Should not iterate over any documents"
        raise AssertionError(msg)


def test_extraction_context_iter_non_iterable():
    """Test non-iterable ExtractionContext iteration"""
    test_doc = PDFDocument([])
    for x in ExtractionContext(test_doc):
        assert isinstance(x, PDFDocument)
        assert x is test_doc


@pytest.mark.parametrize(
    "test_input", (("a", "b"), ["a", "b"], {"a": 1, "b": 2})
)
def test_extraction_context_iter_sequence(test_input):
    """Test non-sequence ExtractionContext iteration"""
    for x, y in zip(ExtractionContext(test_input), test_input, strict=True):
        assert x == y


def test_extraction_context_set():
    """Test non-sequence ExtractionContext iteration"""
    test_input = {"a", "b"}
    test = ExtractionContext(test_input)
    assert set(test) == test_input


def test_extraction_context_text_empty():
    """Test text property with empty context"""
    ctx = ExtractionContext()
    assert not ctx.text


def test_extraction_context_text_single_doc():
    """Test text property with single document"""
    doc = PDFDocument(["page one", "page two"])
    ctx = ExtractionContext(doc)
    assert ctx.text == "page one\npage two"


def test_extraction_context_text_multiple_docs():
    """Test text property concatenates multiple documents"""
    doc1 = PDFDocument(["doc1 page1", "doc1 page2"])
    doc2 = HTMLDocument(["<p>doc2 content</p>"])
    ctx = ExtractionContext([doc1, doc2])
    expected = "doc1 page1\ndoc1 page2\n\ndoc2 content\n\n"
    assert ctx.text == expected


def test_extraction_context_pages_empty():
    """Test pages property with empty context"""
    ctx = ExtractionContext()
    assert ctx.pages == []


def test_extraction_context_pages_single_doc():
    """Test pages property with single document"""
    doc = PDFDocument(["page 1", "page 2", "page 3"])
    ctx = ExtractionContext(doc)
    assert ctx.pages == ["page 1", "page 2", "page 3"]


def test_extraction_context_pages_multiple_docs():
    """Test pages property flattens multiple documents"""
    doc1 = PDFDocument(["doc1 p1", "doc1 p2"])
    doc2 = PDFDocument(["doc2 p1"])
    doc3 = HTMLDocument(["doc3 content"])
    ctx = ExtractionContext([doc1, doc2, doc3])
    assert ctx.pages == ["doc1 p1", "doc1 p2", "doc2 p1", "doc3 content"]


def test_extraction_context_num_documents():
    """Test num_documents property"""
    assert ExtractionContext().num_documents == 0
    assert ExtractionContext(PDFDocument([])).num_documents == 1
    doc_list = [PDFDocument([]), HTMLDocument([""]), PDFDocument([])]
    assert ExtractionContext(doc_list).num_documents == 3


def test_extraction_context_documents_getter():
    """Test documents property getter"""
    doc1 = PDFDocument(["test"])
    doc2 = HTMLDocument(["html"])
    ctx = ExtractionContext([doc1, doc2])
    assert ctx.documents == [doc1, doc2]
    assert ctx.documents[0] is doc1
    assert ctx.documents[1] is doc2


def test_extraction_context_documents_setter():
    """Test documents property setter"""
    ctx = ExtractionContext()
    assert ctx.documents == []

    doc = PDFDocument(["page"])
    ctx.documents = doc
    assert ctx.documents == [doc]

    doc_list = [PDFDocument([]), HTMLDocument([])]
    ctx.documents = doc_list
    assert ctx.documents == doc_list


def test_extraction_context_data_docs_getter():
    """Test data_docs property getter"""
    ctx = ExtractionContext()
    assert ctx.data_docs == []

    ctx._data_docs = [PDFDocument([])]
    assert len(ctx.data_docs) == 1


def test_extraction_context_data_docs_setter_valid():
    """Test data_docs property setter with valid list"""
    ctx = ExtractionContext()
    doc_list = [PDFDocument([]), HTMLDocument([])]
    ctx.data_docs = doc_list
    assert ctx.data_docs == doc_list


def test_extraction_context_data_docs_setter_invalid():
    """Test data_docs property setter raises for non-list"""
    ctx = ExtractionContext()

    with pytest.raises(COMPASSTypeError, match="must be set to a \\*list\\*"):
        ctx.data_docs = PDFDocument([])

    with pytest.raises(COMPASSTypeError, match="must be set to a \\*list\\*"):
        ctx.data_docs = {"doc": PDFDocument([])}

    with pytest.raises(COMPASSTypeError, match="must be set to a \\*list\\*"):
        ctx.data_docs = (PDFDocument([]),)


def test_extraction_context_str_empty():
    """Test string representation of empty context"""
    ctx = ExtractionContext()
    result = str(ctx)
    assert "ExtractionContext with 0 documents" in result
    assert "Registered Data Source Documents: None" in result
    assert "Attrs: None" in result


def test_extraction_context_str_single_doc():
    """Test string representation with single document"""
    doc = PDFDocument(["test"])
    doc.attrs["source"] = "http://example.com/doc.pdf"
    ctx = ExtractionContext(doc)
    result = str(ctx)
    assert "ExtractionContext with 1 document" in result
    assert "http://example.com/doc.pdf" in result


def test_extraction_context_str_multiple_docs():
    """Test string representation with multiple documents"""
    doc1 = PDFDocument(["test"])
    doc1.attrs["source"] = "source1.pdf"
    doc2 = HTMLDocument(["html"])
    doc2.attrs["source"] = "source2.html"
    ctx = ExtractionContext([doc1, doc2])
    result = str(ctx)
    assert "ExtractionContext with 2 documents" in result
    assert "source1.pdf" in result
    assert "source2.html" in result


def test_extraction_context_str_with_data_docs():
    """Test string representation with registered data docs"""
    doc1 = PDFDocument(["test"])
    doc1.attrs["source"] = "main.pdf"
    ctx = ExtractionContext(doc1)

    data_doc = PDFDocument(["data"])
    data_doc.attrs["source"] = "data_source.pdf"
    ctx.data_docs = [data_doc]

    result = str(ctx)
    assert "Registered Data Source Documents:" in result
    assert "data_source.pdf" in result


def test_extraction_context_str_with_attrs():
    """Test string representation with attributes"""
    attrs = {"jurisdiction": "Test County", "year": 2025}
    ctx = ExtractionContext(attrs=attrs)
    result = str(ctx)
    assert "Attrs:" in result
    assert "jurisdiction" in result
    assert "Test County" in result
    assert "year" in result
    assert "2025" in result


def test_extraction_context_str_with_dataframe_attr():
    """Test string representation with DataFrame attribute"""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    ctx = ExtractionContext(attrs={"table_data": df})
    result = str(ctx)
    assert "Attrs:" in result
    assert "table_data" in result
    assert "DataFrame with 3 rows" in result


def test_extraction_context_str_with_multiline_attr():
    """Test string representation with multiline attribute"""
    long_text = "Line 1\nLine 2\nLine 3\n" + "x" * 200
    ctx = ExtractionContext(attrs={"long_text": long_text})
    result = str(ctx)
    assert "Attrs:" in result
    assert "long_text" in result
    assert len(result) < len(long_text) + 100


def test_extraction_context_len():
    """Test __len__ returns number of documents"""
    assert len(ExtractionContext()) == 0
    assert len(ExtractionContext(PDFDocument([]))) == 1
    assert len(ExtractionContext([PDFDocument([]), HTMLDocument([])])) == 2


def test_extraction_context_getitem():
    """Test __getitem__ indexing"""
    doc1 = PDFDocument(["first"])
    doc2 = HTMLDocument(["second"])
    ctx = ExtractionContext([doc1, doc2])

    assert ctx[0] is doc1
    assert ctx[1] is doc2
    assert ctx[-1] is doc2


def test_extraction_context_bool():
    """Test __bool__ conversion"""
    assert not ExtractionContext()
    assert not ExtractionContext(None)
    assert ExtractionContext(PDFDocument([]))
    assert ExtractionContext([PDFDocument([])])


@pytest.mark.asyncio
async def test_mark_doc_as_data_source_no_file_move():
    """Test marking document without file moving"""
    ctx = ExtractionContext()
    doc = PDFDocument(["test content"])
    doc.attrs["source"] = "test.pdf"

    await ctx.mark_doc_as_data_source(doc)

    assert doc in ctx.data_docs
    assert len(ctx.data_docs) == 1
    assert "out_fp" not in doc.attrs


@pytest.mark.asyncio
async def test_mark_doc_as_data_source_with_file_move(monkeypatch, tmp_path):
    """Test marking document with file moving"""
    out_file = tmp_path / "output.pdf"

    async def fake_file_mover(doc_arg, out_fn):  # noqa
        assert out_fn == "output.pdf"
        return out_file

    monkeypatch.setattr(FileMover, "call", fake_file_mover)

    ctx = ExtractionContext()
    doc = PDFDocument(["test content"])
    doc.attrs["source"] = "test.pdf"

    await ctx.mark_doc_as_data_source(doc, out_fn="output.pdf")

    assert doc in ctx.data_docs
    assert len(ctx.data_docs) == 1
    assert doc.attrs["out_fp"] == out_file


@pytest.mark.asyncio
async def test_move_file_to_out_dir(monkeypatch, tmp_path):
    """Test _move_file_to_out_dir helper"""
    output_path = tmp_path / "moved.pdf"

    async def fake_mover(doc_arg, out_fn):  # noqa
        assert out_fn == "output_name.pdf"
        return output_path

    monkeypatch.setattr(FileMover, "call", fake_mover)

    doc = PDFDocument(["content"])
    doc.attrs["source"] = "original.pdf"

    result = await _move_file_to_out_dir(doc, "output_name.pdf")

    assert result is doc
    assert doc.attrs["out_fp"] == output_path


@pytest.mark.parametrize(
    "input_val,expected",
    [
        (None, []),
        (PDFDocument([]), [PDFDocument([])]),
        ([PDFDocument([])], [PDFDocument([])]),
        (["a", "b", "c"], ["a", "b", "c"]),
        (("x", "y"), ["x", "y"]),
        ({"key": "value"}, [{"key": "value"}]),
    ],
)
def test_as_list_conversions(input_val, expected):
    """Test _as_list helper with various inputs"""
    result = _as_list(input_val)
    assert isinstance(result, list)
    if input_val is None:
        assert result == []
    elif isinstance(input_val, (list, tuple)):
        assert result == list(input_val)
    else:
        assert len(result) == 1


def test_as_list_preserves_type():
    """Test _as_list converts to list properly"""
    result = _as_list(("a", "b"))
    assert isinstance(result, list)
    assert not isinstance(result, tuple)


def test_data_docs_repr_empty():
    """Test _data_docs_repr with empty list"""
    result = _data_docs_repr([])
    assert result == "Registered Data Source Documents: None"


def test_data_docs_repr_with_docs():
    """Test _data_docs_repr with documents"""
    doc1 = PDFDocument(["test"])
    doc1.attrs["source"] = "source1.pdf"
    doc2 = HTMLDocument(["html"])
    doc2.attrs["source"] = "source2.html"

    result = _data_docs_repr([doc1, doc2])
    assert "Registered Data Source Documents:" in result
    assert "source1.pdf" in result
    assert "source2.html" in result


def test_data_docs_repr_missing_source():
    """Test _data_docs_repr with missing source attribute"""
    doc = PDFDocument(["test"])
    result = _data_docs_repr([doc])
    assert "Unknown source" in result


def test_attrs_repr_empty():
    """Test _attrs_repr with empty dict"""
    result = _attrs_repr({})
    assert result == "Attrs: None"


def test_attrs_repr_simple_values():
    """Test _attrs_repr with simple key-value pairs"""
    attrs = {"jurisdiction": "Test County", "year": 2025, "active": True}
    result = _attrs_repr(attrs)
    assert "Attrs:" in result
    assert "jurisdiction" in result
    assert "Test County" in result
    assert "year" in result
    assert "2025" in result
    assert "active" in result


def test_attrs_repr_with_dataframe():
    """Test _attrs_repr formats DataFrames"""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    attrs = {"my_table": df}
    result = _attrs_repr(attrs)
    assert "Attrs:" in result
    assert "my_table" in result
    assert "DataFrame with 5 rows" in result


def test_attrs_repr_with_multiline_string():
    """Test _attrs_repr shortens multiline strings"""
    long_text = "\n".join([f"Line {i}" for i in range(50)])
    attrs = {"description": long_text}
    result = _attrs_repr(attrs)
    assert "Attrs:" in result
    assert "description" in result
    assert len(result) < len(long_text) + 50


def test_attrs_repr_formatting_alignment():
    """Test _attrs_repr aligns values properly"""
    attrs = {"short": "val", "very_long_key": "value2"}
    result = _attrs_repr(attrs)
    lines = result.split("\n")[1:]
    assert len(lines) == 2
    assert "\t" in lines[0]
    assert "\t" in lines[1]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
