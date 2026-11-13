"""Tests for skipping the legal-text check during relevance parsing"""

from types import SimpleNamespace

import pytest
from elm.web.document import PDFDocument

from compass.extraction import apply


class Collector:
    """Text collector that keeps every chunk"""

    OUT_LABEL = "relevant_text"

    def __init__(self, **_):
        self.relevant_text = ""

    async def check_chunk(self, chunk_parser, ind):
        self.relevant_text += chunk_parser.text_chunks[ind]
        return True


@pytest.fixture
def validators(monkeypatch):
    """Record the legal-text validator handed to chunk parsing"""
    seen = []

    async def parse_by_chunks(chunk_parser, heuristic, validator, **kwargs):
        seen.append(validator)
        for callback in kwargs["callbacks"]:
            await callback(chunk_parser, 0)

    monkeypatch.setattr(apply, "parse_by_chunks", parse_by_chunks)
    monkeypatch.setattr(
        apply, "LegalTextValidator", lambda **_: "legal-text-validator"
    )
    return seen


@pytest.mark.parametrize(
    ("attrs", "checked"),
    [
        ({}, True),
        ({"check_if_legal_doc": True}, True),
        ({"check_if_legal_doc": False}, False),
    ],
)
async def test_documents_can_skip_the_legal_text_check(
    validators, attrs, checked
):
    """Documents already confirmed as legal text are not re-checked"""
    doc = PDFDocument(["Data centers require a special use permit."])
    doc.attrs.update(source="https://county.gov/ordinance.pdf", **attrs)
    model_config = SimpleNamespace(
        text_splitter=SimpleNamespace(split_text=lambda text: [text]),
        llm_service=None,
        llm_call_kwargs={},
    )
    found = await apply.check_for_relevant_text(
        doc,
        model_config=model_config,
        heuristic=None,
        tech="data centers",
        text_collectors=[Collector],
    )
    assert found
    assert (validators == ["legal-text-validator"]) is checked
    assert (validators == [None]) is not checked
