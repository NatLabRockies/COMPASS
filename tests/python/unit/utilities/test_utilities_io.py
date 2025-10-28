"""Test COMPASS I/O utilities"""

import os
from pathlib import Path

import pytest

from compass.utilities.io import load_local_docs
from compass.services.cpu import (
    PDFLoader,
    OCRPDFLoader,
    read_pdf_file,
    read_pdf_file_ocr,
)
from compass.services.provider import RunningAsyncServices
from compass.services.threaded import read_html_file, HTMLFileLoader
from compass.exceptions import COMPASSNotInitializedError


PYT_CMD = os.getenv("TESSERACT_CMD")


@pytest.mark.asyncio
async def test_basic_load_pdf(test_data_files_dir):
    """Test basic loading of local PDF document"""
    test_fp = test_data_files_dir / "Caneadea New York.pdf"

    docs = await load_local_docs([test_fp])
    assert len(docs) == 1
    doc = docs[0]
    assert not doc.empty
    assert Path(doc.attrs.get("source_fp")) == test_fp
    assert len(doc.pages) == 3


@pytest.mark.asyncio
async def test_basic_load_html(test_data_files_dir):
    """Test basic loading of local HTML document"""
    test_fp = test_data_files_dir / "Whatcom.txt"

    docs = await load_local_docs([test_fp])
    assert len(docs) == 1
    doc = docs[0]
    assert not doc.empty
    assert Path(doc.attrs.get("source_fp")) == test_fp
    assert len(doc.pages) == 1


@pytest.mark.asyncio
async def test_basic_load_pdf_with_service(test_data_files_dir):
    """Test basic loading of local PDF document with service"""
    test_fp = test_data_files_dir / "Caneadea New York.pdf"

    with pytest.raises(
        COMPASSNotInitializedError,
        match=r"Must initialize the queue for 'PDFLoader'.",
    ):
        await read_pdf_file(test_fp)

    async with RunningAsyncServices([PDFLoader()]):
        doc, __ = await read_pdf_file(test_fp)
        doc_2 = await load_local_docs(
            [test_fp], pdf_read_coroutine=read_pdf_file
        )

    assert not doc.empty
    assert not doc_2[0].empty
    assert doc.text == doc_2[0].text


@pytest.mark.skipif(
    not PYT_CMD, reason="requires PyTesseract command to be set"
)
@pytest.mark.asyncio
async def test_basic_load_ocr_pdf_with_service(test_data_files_dir):
    """Test basic loading of local PDF document with service"""
    import pytesseract  # noqa: PLC0415

    pytesseract.pytesseract.tesseract_cmd = PYT_CMD

    test_fp = test_data_files_dir / "Sedgwick Kansas.pdf"

    with pytest.raises(
        COMPASSNotInitializedError,
        match=r"Must initialize the queue for 'OCRPDFLoader'.",
    ):
        await read_pdf_file_ocr(test_fp)

    async with RunningAsyncServices([OCRPDFLoader()]):
        doc, __ = await read_pdf_file_ocr(test_fp)
        doc_2 = await load_local_docs(
            [test_fp], pdf_ocr_read_coroutine=read_pdf_file_ocr
        )

    assert not doc.empty
    assert not doc_2[0].empty
    assert doc.text == doc_2[0].text


@pytest.mark.asyncio
async def test_basic_load_html_with_service(test_data_files_dir):
    """Test basic loading of local HTML document with service"""
    test_fp = test_data_files_dir / "Whatcom.txt"

    with pytest.raises(
        COMPASSNotInitializedError,
        match=r"Must initialize the queue for 'HTMLFileLoader'.",
    ):
        await read_html_file(test_fp)

    async with RunningAsyncServices([HTMLFileLoader()]):
        doc, __ = await read_html_file(test_fp)
        doc_2 = await load_local_docs(
            [test_fp], html_read_coroutine=read_html_file
        )

    assert not doc.empty
    assert not doc_2[0].empty
    assert doc.text == doc_2[0].text


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
