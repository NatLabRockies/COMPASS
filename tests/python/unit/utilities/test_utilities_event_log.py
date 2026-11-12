"""Tests for phase-organized jurisdiction event logs"""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from compass.llm.calling import ChatLLMCaller, SchemaOutputLLMCaller
from compass.utilities.event_log import (
    document_summary,
    EventLog,
    body_preview,
    emit_event,
    event_phase,
    event_tags,
    loggable_headers,
    token_costs,
    use_event_log,
)


def read_events(path):
    """Load every event from a JSONL log"""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]


def api_response(content="{}", finish_reason="stop"):
    """Build a minimal chat-completions response"""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=5,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
        ),
    )


class FakeService:
    """LLM service that reports usage the way the OpenAI service does"""

    model_name = "test-model"

    def __init__(self, content='{"answer": 1}'):
        self.content = content
        self.calls = []

    async def call(self, usage_tracker=None, usage_sub_label=None, **kwargs):
        self.calls.append(kwargs)
        if usage_tracker is not None:
            usage_tracker.update_from_model(
                model=self.model_name,
                response=api_response(self.content),
                sub_label=usage_sub_label,
            )
        return self.content


class Tracker:
    """Usage tracker that records updates"""

    def __init__(self):
        self.updates = []

    def update_from_model(self, model=None, response=None, sub_label=None):
        self.updates.append((model, sub_label))


def test_phases_nest_and_tag_events(tmp_path):
    """Events carry the innermost phase; markers record timing"""
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        emit_event("before")
        with event_phase("collect", step="search_engine"):
            with event_phase("search"):
                emit_event("inside")
            emit_event("back_in_collect")
    events = read_events(log.path)
    assert [(e["phase"], e["type"]) for e in events] == [
        (None, "before"),
        ("collect", "phase_started"),
        ("search", "phase_started"),
        ("search", "inside"),
        ("search", "phase_finished"),
        ("collect", "back_in_collect"),
        ("collect", "phase_finished"),
    ]
    assert events[1]["step"] == "search_engine"
    assert events[2]["parent_phase"] == "collect"
    assert "seconds" in events[4]


def test_phase_records_errors(tmp_path):
    """A failing phase is recorded before the error propagates"""
    log = EventLog(tmp_path / "events.jsonl")
    with (
        use_event_log(log),
        pytest.raises(ValueError, match="boom"),
        event_phase("extract"),
    ):
        raise ValueError("boom")
    assert read_events(log.path)[-1]["error"] == "ValueError: boom"


def test_without_a_log_events_are_ignored(tmp_path):
    """Pipeline code can emit freely when no log is active"""
    emit_event("ignored")
    with event_phase("filter"):
        emit_event("ignored")
    assert not list(tmp_path.iterdir())


async def test_llm_calls_are_recorded_with_api_details(tmp_path):
    """Exact messages, response, and API metadata are logged"""
    log = EventLog(tmp_path / "events.jsonl")
    tracker = Tracker()
    caller = SchemaOutputLLMCaller(FakeService(), usage_tracker=tracker)
    with use_event_log(log), event_phase("filter"):
        out = await caller.call(
            sys_msg="system",
            content="document text",
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "chunk_validation"},
            },
            usage_sub_label="document_content_validation",
        )
    assert out == {"answer": 1}
    assert tracker.updates == [("test-model", "document_content_validation")]
    events = read_events(log.path)
    (schema,) = [e for e in events if e["type"] == "llm_schema"]
    request, response = [
        e for e in events if e["type"] in {"llm_request", "llm_response"}
    ]
    assert request["schema_id"] == schema["schema_id"]
    assert schema["response_format"]["json_schema"]["name"] == (
        "chunk_validation"
    )
    assert request["phase"] == "filter"
    assert request["purpose"] == "document_content_validation"
    assert request["response_format"] == "chunk_validation"
    assert request["messages"][1]["content"] == "document text"
    assert response["call"] == request["call"]
    assert response["response"] == '{"answer": 1}'
    assert response["api"]["finish_reason"] == "stop"
    assert response["api"]["reasoning_tokens"] == 2


async def test_chat_logs_only_the_newest_turn(tmp_path):
    """Long decision-tree chats do not repeat their history"""
    log = EventLog(tmp_path / "events.jsonl")
    caller = ChatLLMCaller(FakeService("Yes."), system_message="system")
    with use_event_log(log):
        await caller.call("first question")
        await caller.call("second question")
    requests = [e for e in read_events(log.path) if e["type"] == "llm_request"]
    assert requests[1]["messages"] == [
        {"role": "user", "content": "second question"}
    ]
    assert requests[1]["history_messages"] == 3
    # the first turn is logged whole, so the system message is on record
    assert requests[0]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first question"},
    ]
    assert requests[0]["history_messages"] == 0


async def test_llm_parameters_redact_header_credentials(tmp_path):
    """Credentials passed as request headers stay out of the log"""
    log = EventLog(tmp_path / "events.jsonl")
    caller = SchemaOutputLLMCaller(
        FakeService(),
        seed=42,
        extra_headers={"Authorization": "Bearer secret", "X-Team": "ords"},
    )
    with use_event_log(log):
        await caller.call("system", "text", response_format={})
    (request,) = [
        e for e in read_events(log.path) if e["type"] == "llm_request"
    ]
    assert request["parameters"] == {
        "seed": 42,
        "extra_headers": {"authorization": "<redacted>", "x-team": "ords"},
    }
    assert "secret" not in log.path.read_text()


def test_document_summary_uses_backend_format_and_page_count():
    """A PDF a backend converted to text is still described as a PDF"""
    doc = SimpleNamespace(
        attrs={"source": "a.pdf", "doc_type": "pdf", "num_pages": 12},
        pages=["all the text"],
        text="all the text",
    )
    assert document_summary(doc) == {
        "source": "a.pdf",
        "pdf": True,
        "pages": 12,
        "characters": 12,
    }


def test_nan_values_are_written_as_null(tmp_path):
    """Empty pandas cells (NaN) must not produce invalid JSON lines"""
    log = EventLog(tmp_path / "events.jsonl")
    log.emit("rows", rows=[{"value": float("nan"), "n": float("inf")}])
    line = log.path.read_text(encoding="utf-8").strip()
    assert "NaN" not in line
    assert json.loads(line)["rows"] == [{"value": None, "n": None}]


async def test_chat_turns_share_a_conversation(tmp_path):
    """Decision-tree turns are grouped; separate callers are not"""
    log = EventLog(tmp_path / "events.jsonl")
    chat = ChatLLMCaller(FakeService("Yes."), system_message="system")
    single = SchemaOutputLLMCaller(FakeService())
    with use_event_log(log):
        await chat.call("first")
        await chat.call("second")
        await single.call("system", "text", response_format={})
        await single.call("system", "text", response_format={})
    requests = [e for e in read_events(log.path) if e["type"] == "llm_request"]
    conversations = [e["conversation"] for e in requests]
    assert conversations[0] == conversations[1]
    assert len(set(conversations)) == 3


def test_tags_are_attached_and_nest(tmp_path):
    """Tags identify the document and decision tree behind an event"""
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log), event_tags(document="https://a.gov/doc.pdf"):
        with event_tags(thread="Correct jurisdiction type"):
            emit_event("inner")
        emit_event("outer")
    emit_event("untagged")
    inner, outer = read_events(log.path)
    assert inner["tags"] == {
        "document": "https://a.gov/doc.pdf",
        "thread": "Correct jurisdiction type",
    }
    assert outer["tags"] == {"document": "https://a.gov/doc.pdf"}


def test_token_costs_use_model_rates():
    """Costs follow the registry's per-million-token rates"""
    costs = token_costs(
        "gpt-5.6-luna",
        prompt_tokens=99_077,
        completion_tokens=1_000,
        reasoning_tokens=400,
    )
    assert costs["prompt_cost"] == pytest.approx(99_077 * 0.2 / 1e6)
    assert costs["completion_cost"] == pytest.approx(1_000 * 1.2 / 1e6)
    assert costs["reasoning_cost"] == pytest.approx(400 * 1.2 / 1e6)
    assert costs["total_cost"] == pytest.approx(
        costs["prompt_cost"] + costs["completion_cost"]
    )
    assert token_costs("unknown-model", prompt_tokens=10)["total_cost"] is None


def test_loggable_headers_redact_secrets_but_keep_cookie_names():
    """Credentials never reach the log; cookie names still do"""
    headers = httpx.Headers(
        [
            ("Authorization", "Bearer secret"),
            ("Set-Cookie", "session=abc; path=/"),
            ("Set-Cookie", "challenge=xyz"),
            ("Server", "Sucuri/Cloudproxy"),
        ]
    )
    assert loggable_headers(headers) == {
        "authorization": "<redacted>",
        "set-cookie": "session=<redacted>, challenge=<redacted>",
        "server": "Sucuri/Cloudproxy",
    }
    assert loggable_headers(None) == {}
    assert loggable_headers({"Cookie": "a=1; b=2"}) == {
        "cookie": "a=<redacted>; b=<redacted>"
    }


def test_loggable_headers_split_browser_style_set_cookie():
    """Browsers join repeated Set-Cookie headers with newlines"""
    headers = {"set-cookie": "sucuri_uuid=abc; path=/\ntracker=1; Secure"}
    assert loggable_headers(headers) == {
        "set-cookie": "sucuri_uuid=<redacted>, tracker=<redacted>"
    }


def test_body_preview_shows_text_but_not_documents():
    """Text bodies get a preview; PDFs and binaries do not"""
    assert body_preview(b"<html>Blocked</html>", "text/html") == (
        "<html>Blocked</html>"
    )
    assert body_preview(b"%PDF-1.7 ...", "application/pdf") is None
    assert body_preview(b"\x00\x01", "application/octet-stream") is None
    assert len(body_preview(b"x" * 5000, "text/plain")) == 2000
