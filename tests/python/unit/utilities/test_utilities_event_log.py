"""Tests for phase-organized jurisdiction event logs"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from compass.llm.calling import (
    PLAIN_TEXT_INSTRUCTIONS,
    ChatLLMCaller,
    LLMCaller,
    SchemaOutputLLMCaller,
)
from compass.services.base import Service
from compass.services.provider import RunningAsyncServices
from compass.utilities.event_log import (
    event_step,
    error_details,
    document_summary,
    EventLog,
    body_preview,
    emit_event,
    event_tags,
    loggable_headers,
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


def _retries_exhausted():
    """The error a call gives when every retry lost its connection

    As httpcore does, the read error is re-raised ``from None``, which
    keeps the socket error only as its hidden context.
    """
    reset = ConnectionResetError(54, "Connection reset by peer")
    read = httpx.ReadError("")
    read.__context__, read.__suppress_context__ = reset, True
    connection = ConnectionError("Connection error.")
    connection.__cause__ = read
    exhausted = RuntimeError("Maximum number of retries (3) exceeded")
    exhausted.__cause__ = connection
    return exhausted


def test_error_details_follow_the_cause_chain():
    """An error re-raised by a retry wrapper keeps what caused it"""
    out = error_details(_retries_exhausted())
    assert out["error"] == (
        "RuntimeError: Maximum number of retries (3) exceeded "
        "(caused by ConnectionResetError: [Errno 54] Connection reset by peer)"
    )
    assert out["error_chain"] == [
        {
            "type": "RuntimeError",
            "message": "Maximum number of retries (3) exceeded",
        },
        {"type": "ConnectionError", "message": "Connection error."},
        {"type": "httpx.ReadError", "message": ""},
        {
            "type": "ConnectionResetError",
            "message": "[Errno 54] Connection reset by peer",
        },
    ]
    assert error_details(ValueError("boom")) == {"error": "ValueError: boom"}


def test_error_details_record_where_the_error_was_raised():
    """Each error in the chain lists the frames it was raised through"""

    def fail():
        raise ValueError("boom")

    try:
        fail()
    except ValueError as error:
        out = error_details(error)
    frame = out["error_chain"][0]["frames"][-1]
    assert (
        frame["file"]
        == "tests/python/unit/utilities/test_utilities_event_log.py"
    )
    assert frame["function"] == "fail"
    assert frame["line"] == fail.__code__.co_firstlineno + 1


def test_error_details_include_the_code_that_led_to_the_handler():
    """An error logged where it is caught still shows who called that code"""

    def handler():
        try:
            raise ValueError("boom")
        except ValueError as error:
            return error_details(error)

    def caller():
        return handler()

    frames = caller()["error_chain"][0]["frames"]
    names = [f["function"] for f in frames]
    assert names[-2:] == ["caller", "handler"]
    assert "error_details" not in names


async def test_failed_llm_call_records_the_underlying_error(tmp_path):
    """A call that finally fails logs why, not only that retries ran out"""

    class FailingService(FakeService):
        async def call(self, **_):
            raise _retries_exhausted()

    log = EventLog(tmp_path / "events.jsonl")
    caller = SchemaOutputLLMCaller(FailingService())
    with use_event_log(log), pytest.raises(RuntimeError):
        await caller.call("system", "text", response_format={})
    response = read_events(log.path)[-1]
    assert response["type"] == "llm_response"
    assert "Connection reset by peer" in response["error"]
    assert response["error_chain"][1]["message"] == "Connection error."


class _RetryingService(Service):
    """Runs each call as a queued job that retries once before answering"""

    model_name = "test-model"

    @property
    def can_process(self):
        return True

    async def process(self, **_):
        try:
            raise httpx.ReadError("")
        except httpx.ReadError as error:
            logging.getLogger("elm.utilities.retry").info(
                "Error: %s. Retrying in %.2f seconds.", error, 1.5
            )
        return "answer"


async def test_retries_are_recorded_with_their_call(tmp_path):
    """A retry inside a service job lands in the caller's event log"""
    service = _RetryingService()
    log = EventLog(tmp_path / "events.jsonl")
    async with RunningAsyncServices([service]):
        with use_event_log(log):
            assert await LLMCaller(service).call("system", "text") == "answer"
    (retry,) = [e for e in read_events(log.path) if e["type"] == "llm_retry"]
    request = next(
        e for e in read_events(log.path) if e["type"] == "llm_request"
    )
    assert retry["tags"]["call"] == request["call"]
    assert retry["by"] == "retry wrapper"
    assert retry["delay"] == pytest.approx(1.5)
    assert retry["error"] == "ReadError"


def test_retries_outside_llm_calls_are_download_retries(tmp_path):
    """Retries outside an LLM call (e.g. downloads) are not LLM retries"""
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        try:
            raise httpx.ReadError("")
        except httpx.ReadError as error:
            logging.getLogger("elm.utilities.retry").info(
                "Error: %s. Retrying in %.2f seconds.", error, 4.0
            )
    (retry,) = [
        e for e in read_events(log.path) if e["type"].endswith("retry")
    ]
    assert retry["type"] == "download_retry"
    assert retry["delay"] == pytest.approx(4.0)


def test_without_a_log_events_are_ignored(tmp_path):
    """Pipeline code can emit freely when no log is active"""
    emit_event("ignored")
    with event_step("filter"):
        emit_event("ignored")
    assert not list(tmp_path.iterdir())


async def test_llm_calls_are_recorded_with_api_details(tmp_path):
    """Exact messages, response, and API metadata are logged"""
    log = EventLog(tmp_path / "events.jsonl")
    tracker = Tracker()
    caller = SchemaOutputLLMCaller(FakeService(), usage_tracker=tracker)
    with use_event_log(log), event_step("filter"):
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
    assert request["tags"]["step"] == "filter"
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
        {
            "role": "system",
            "content": f"system\n\n{PLAIN_TEXT_INSTRUCTIONS}",
        },
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
