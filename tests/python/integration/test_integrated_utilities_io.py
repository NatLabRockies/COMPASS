"""COMPASS I/O tests"""

import os
from pathlib import Path

import pytest
from elm.web.search.run import load_docs

from compass.services.cpu import FileLoader, read_docling_local_file
from compass.web.file_loader import AsyncLocalDoclingFileLoader
from compass.services.provider import RunningAsyncServices
from compass.exceptions import COMPASSNotInitializedError


PYT_CMD = os.getenv("TESSERACT_CMD")


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Docling too heavy for GHA"
)
@pytest.mark.asyncio
async def test_basic_load_pdf(test_data_files_dir):
    """Test basic loading of local PDF document"""
    test_fp = test_data_files_dir / "Caneadea New York.pdf"

    fl = AsyncLocalDoclingFileLoader()
    async with RunningAsyncServices([FileLoader()]):
        docs = await load_docs([test_fp], fl)

    assert len(docs) == 1
    doc = docs[0]
    assert not doc.empty
    assert Path(doc.attrs.get("source_fp")) == test_fp
    assert len(doc.pages) == 1


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Docling too heavy for GHA"
)
@pytest.mark.asyncio
async def test_basic_load_html(test_data_files_dir):
    """Test basic loading of local HTML document"""
    test_fp = test_data_files_dir / "Whatcom.txt"

    fl = AsyncLocalDoclingFileLoader()
    async with RunningAsyncServices([FileLoader()]):
        docs = await load_docs([test_fp], fl)

    assert len(docs) == 1
    doc = docs[0]
    assert not doc.empty
    assert Path(doc.attrs.get("source_fp")) == test_fp
    assert len(doc.pages) == 1


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Docling too heavy for GHA"
)
@pytest.mark.asyncio
async def test_basic_load_pdf_with_service(test_data_files_dir):
    """Test basic loading of local PDF document with service"""
    test_fp = test_data_files_dir / "Caneadea New York.pdf"

    with pytest.raises(
        COMPASSNotInitializedError,
        match=r"Must initialize the queue for 'FileLoader'.",
    ):
        await read_docling_local_file(test_fp)

    fl = AsyncLocalDoclingFileLoader()
    async with RunningAsyncServices([FileLoader()]):
        doc, __ = await read_docling_local_file(test_fp)
        doc_2 = await load_docs([test_fp], fl)

    assert not doc.empty
    assert not doc_2[0].empty
    assert doc.text == doc_2[0].text


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true" or not PYT_CMD,
    reason="requires PyTesseract command to be set; Docling too heavy for GHA",
)
@pytest.mark.asyncio
async def test_basic_load_ocr_pdf_with_service(test_data_files_dir):
    """Test basic loading of local PDF document with service"""
    test_fp = test_data_files_dir / "Sedgwick Kansas.pdf"

    with pytest.raises(
        COMPASSNotInitializedError,
        match=r"Must initialize the queue for 'FileLoader'.",
    ):
        await read_docling_local_file(test_fp)

    fl = AsyncLocalDoclingFileLoader(pytesseract_exe_fp=PYT_CMD)
    async with RunningAsyncServices([FileLoader()]):
        doc, __ = await read_docling_local_file(
            test_fp, pytesseract_exe_fp=PYT_CMD
        )
        doc_2 = await load_docs([test_fp], fl)

    assert not doc.empty
    assert not doc_2[0].empty
    assert doc.text == doc_2[0].text


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Docling too heavy for GHA"
)
@pytest.mark.asyncio
async def test_basic_load_html_with_service(test_data_files_dir):
    """Test basic loading of local HTML document with service"""
    test_fp = test_data_files_dir / "Whatcom.txt"

    with pytest.raises(
        COMPASSNotInitializedError,
        match=r"Must initialize the queue for 'FileLoader'.",
    ):
        await read_docling_local_file(test_fp)

    fl = AsyncLocalDoclingFileLoader()
    async with RunningAsyncServices([FileLoader()]):
        doc, __ = await read_docling_local_file(test_fp)
        doc_2 = await load_docs([test_fp], fl)

    assert not doc.empty
    assert not doc_2[0].empty
    assert doc.text == doc_2[0].text


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
