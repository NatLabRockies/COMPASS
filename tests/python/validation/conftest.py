"""Fixtures to use for validation tests"""

import os
import asyncio
from functools import partial

import pytest
import openai
from elm import ApiBase
from elm.web.document import PDFDocument, HTMLDocument
from elm.utilities.parse import read_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

from compass.utilities import RTS_SEPARATORS
from compass.services.openai import OpenAIService
from compass.services.provider import RunningAsyncServices


TESTING_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    RTS_SEPARATORS,
    chunk_size=3000,
    chunk_overlap=300,
    length_function=partial(ApiBase.count_tokens, model="gpt-4"),
    is_separator_regex=True,
)


@pytest.fixture(scope="session")
def event_loop():
    """Override default event loop fixture to make it module-level"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def oai_async_azure_client():
    """OpenAi Azure client to use for tests"""
    if os.getenv("AZURE_OPENAI_API_KEY") is None:
        return None

    return openai.AsyncAzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_VERSION"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    )


@pytest.fixture(scope="session")
def oai_llm_service(oai_async_azure_client):
    """OpenAi Azure client to use for tests"""
    model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini")
    return OpenAIService(
        client=oai_async_azure_client, model_name=model_name, rate_limit=1e6
    )


@pytest.fixture(scope="session", autouse=True)
async def running_openai_service(oai_llm_service):
    """Set up running OpenAI service to use for tests"""
    if os.getenv("AZURE_OPENAI_API_KEY") is None:
        yield
    else:
        async with RunningAsyncServices([oai_llm_service]):
            yield


@pytest.fixture(scope="session")
def text_splitter():
    """Text splitter to uses for tests"""
    return TESTING_TEXT_SPLITTER


@pytest.fixture(scope="session")
def doc_loader(test_data_dir):
    """Text splitter to uses for tests"""
    return partial(_load_doc, test_data_dir)


def _load_doc(test_data_dir, doc_fn):
    """Load PDF or HTML doc for tests"""
    doc_fp = test_data_dir / doc_fn
    if doc_fp.suffix == ".pdf":
        with doc_fp.open("rb") as fh:
            pages = read_pdf(fh.read())
            return PDFDocument(pages)

    with doc_fp.open("r", encoding="utf-8") as fh:
        text = fh.read()
        return HTMLDocument([text], text_splitter=TESTING_TEXT_SPLITTER)
