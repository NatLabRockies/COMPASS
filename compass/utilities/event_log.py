"""Replayable, phase-organized event logs for one jurisdiction's run

An :class:`EventLog` appends one JSON object per line to a file as each
step happens, so a run can be followed live and replayed afterwards.
The log for the jurisdiction being processed, and the phase it is in,
are tracked with context variables, so code anywhere in the pipeline can
record events with :func:`emit_event` without passing the log around.

Events link to the workflow map (``compass/data/workflow.json5``): each
names the step it happened in with :func:`event_step`, and each decision
records which way it went with :func:`emit_branch`, so a replay can draw
the path a run took against every path that was possible.
"""

import json
import math
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path

from compass.utilities.costs import LLM_COST_REGISTRY, cost_for_model


_CURRENT_LOG = ContextVar("compass_event_log", default=None)
_CURRENT_PHASE = ContextVar("compass_event_phase", default=None)
_CURRENT_TAGS = ContextVar("compass_event_tags", default=None)


class EventLog:
    """Append-only JSONL event stream for one jurisdiction

    A log opened on a file that already holds events (an earlier step of
    a workflow run as separate search, collect, and extract steps)
    continues it: events are added after the existing ones, their
    sequence numbers and elapsed times carry on, and the IDs given out
    are marked with the run they belong to so they stay unique.
    """

    def __init__(self, path):
        """

        Parameters
        ----------
        path : path-like
            File to write events to. It is created if it does not
            exist, and added to if it does.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        last, runs = _existing_events(self.path)
        self.is_new = last is None
        """bool: Whether this log started a new file"""
        self._count = (last or {}).get("seq", 0)
        self._offset = (last or {}).get("t", 0)
        self._run = runs + 1
        self._ids = {}
        self._schemas = {}
        self._graphs = {}
        self._started = time.monotonic()

    def emit(self, kind, **fields):
        """Append one event, tagged with the current phase"""
        self._count += 1
        record = {
            "seq": self._count,
            "t": round(self._offset + time.monotonic() - self._started, 3),
            "phase": _CURRENT_PHASE.get(),
            "type": kind,
        }
        tags = _CURRENT_TAGS.get()
        if tags:
            record["tags"] = tags
        record.update(fields)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    _json_safe(record),
                    ensure_ascii=False,
                    default=str,
                    allow_nan=False,
                )
            )
            stream.write("\n")

    def next_id(self, prefix):
        """Sequential ID, e.g. linking a request to its response"""
        self._ids[prefix] = self._ids.get(prefix, 0) + 1
        if self._run > 1:
            return f"{prefix}-{self._run}.{self._ids[prefix]}"
        return f"{prefix}-{self._ids[prefix]}"

    def next_call_id(self):
        """ID linking an ``llm_request`` to its ``llm_response``"""
        return self.next_id("llm")

    def schema_id(self, response_format):
        """ID for a structured-output schema, recorded the first time

        Schemas can be large and are often sent with many calls, so each
        distinct one is written once as an ``llm_schema`` event and
        referred to by ID afterwards.
        """
        key = json.dumps(response_format, sort_keys=True, default=str)
        if key not in self._schemas:
            self._schemas[key] = self.next_id("schema")
            self.emit(
                "llm_schema",
                schema_id=self._schemas[key],
                response_format=response_format,
            )
        return self._schemas[key]

    def graph_id(self, description):
        """ID for a decision-tree graph, recorded the first time

        A tree's graph (every question and branch, including those a
        run never reaches) is written once as a ``decision_tree_graph``
        event; trees asking the same questions, such as one per page of
        a document, refer to it by ID.

        Parameters
        ----------
        description : dict
            ``nodes`` and ``edges`` of the graph, as built by
            :func:`compass.common.tree.describe_graph`.
        """
        key = json.dumps(description, sort_keys=True, default=str)
        if key not in self._graphs:
            self._graphs[key] = self.next_id("graph")
            self.emit(
                "decision_tree_graph",
                graph_id=self._graphs[key],
                **description,
            )
        return self._graphs[key]

    @contextmanager
    def phase(self, name, **fields):
        """Record a named phase and tag events emitted inside it"""
        parent = _CURRENT_PHASE.get()
        token = _CURRENT_PHASE.set(name)
        started = time.monotonic()
        self.emit("phase_started", parent_phase=parent, **fields)
        try:
            yield
        except BaseException as exc:
            self.emit(
                "phase_finished",
                seconds=round(time.monotonic() - started, 3),
                **error_details(exc),
            )
            raise
        else:
            self.emit(
                "phase_finished",
                seconds=round(time.monotonic() - started, 3),
            )
        finally:
            _CURRENT_PHASE.reset(token)


def _existing_events(path):
    """The last event already in a log file, and how many runs it holds

    Returns
    -------
    tuple
        The last event (``None`` for a missing or empty file) and the
        number of ``jurisdiction_started`` events, one per run.
    """
    if not path.exists():
        return None, 0
    last, runs = None, 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
            runs += last.get("type") == "jurisdiction_started"
    return last, runs


def error_details(error):
    """Event fields describing an exception and what caused it

    Retry wrappers and client libraries re-raise errors as their own
    (for example, ELM's "Maximum number of retries (3) exceeded" raised
    from the OpenAI client's ``APIConnectionError``, itself raised from
    an ``httpx.ReadError``). The outermost error alone rarely says what
    went wrong, so the whole chain is recorded, down to the operating
    system's error where there is one (such as a connection reset).
    Libraries sometimes re-raise with ``from None``, which hides the
    original error from tracebacks without discarding it; it is still
    followed here, since that original error is usually the useful one.

    Parameters
    ----------
    error : BaseException
        Exception to describe.

    Returns
    -------
    dict
        ``error``: the exception as ``"Type: message"``, followed by
        the innermost cause when there is one. ``error_chain`` (only
        when the exception has a cause): every exception in the chain,
        outermost first, each with its full ``type`` name and
        ``message``.
    """
    chain, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append(error)
        error = error.__cause__ or error.__context__
    text = _error_text(chain[0])
    if len(chain) == 1:
        return {"error": text}
    return {
        "error": f"{text} (caused by {_error_text(chain[-1])})",
        "error_chain": [
            {"type": _error_type(exc), "message": str(exc)} for exc in chain
        ],
    }


def _error_text(error):
    """An exception as ``"Type: message"``, or just its type"""
    message = str(error)
    return (
        f"{type(error).__name__}: {message}"
        if message
        else (type(error).__name__)
    )


def _error_type(error):
    """Full name of an exception's type, e.g. ``httpx.ReadError``"""
    kind = type(error)
    if kind.__module__ == "builtins":
        return kind.__qualname__
    return f"{kind.__module__}.{kind.__qualname__}"


def _json_safe(value):
    """Replace NaN and infinite floats, which JSON cannot represent"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@contextmanager
def use_event_log(log):
    """Make `log` the current event log within this context"""
    token = _CURRENT_LOG.set(log)
    try:
        yield log
    finally:
        _CURRENT_LOG.reset(token)


@contextmanager
def event_tags(**tags):
    """Attach tags (e.g. the document being checked) to events inside"""
    token = _CURRENT_TAGS.set({**(_CURRENT_TAGS.get() or {}), **tags})
    try:
        yield
    finally:
        _CURRENT_TAGS.reset(token)


def event_step(step):
    """Link events inside to a node of the workflow map

    Parameters
    ----------
    step : str
        ID of the node in ``compass/data/workflow.json5`` that the
        events record, stored as the ``step`` tag. Nested steps override
        outer ones.
    """
    return event_tags(step=step)


def emit_branch(step, outcome, **fields):
    """Record which way a workflow decision went

    Parameters
    ----------
    step : str
        ID of the decision's node in ``compass/data/workflow.json5``.
    outcome : str
        The ``when`` label of the edge taken, as listed in the map.
    **fields
        Details explaining the outcome (e.g. a score and threshold).
    """
    with event_tags(step=step):
        emit_event("branch", outcome=outcome, **fields)


def current_event_log():
    """EventLog or None: The event log for the work in progress"""
    return _CURRENT_LOG.get()


def emit_event(kind, **fields):
    """Record an event if an event log is active, else do nothing"""
    log = _CURRENT_LOG.get()
    if log is not None:
        log.emit(kind, **fields)


def event_phase(name, **fields):
    """Context manager for a phase in the current event log, if any"""
    log = _CURRENT_LOG.get()
    return log.phase(name, **fields) if log is not None else nullcontext()


_SECRET_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "x-api-key", "api-key"}
)


def loggable_headers(headers):
    """HTTP headers as a dict for an event, with secrets redacted

    Credentials and cookie values are replaced by ``<redacted>``; cookie
    names are kept, since they often identify the service in the way
    (e.g. a firewall's challenge cookie). Repeated headers are joined
    with ", ".

    Parameters
    ----------
    headers : mapping or None
        Headers from an HTTP client (``httpx`` or ``aiohttp``) or a
        plain dict.

    Returns
    -------
    dict
        Lower-cased header names mapped to their (redacted) values.
    """
    if not headers:
        return {}
    items = (
        headers.multi_items()
        if hasattr(headers, "multi_items")
        else headers.items()
    )
    out = {}
    for key, value in items:
        name = str(key).lower()
        shown = _loggable_header_value(name, value)
        out[name] = f"{out[name]}, {shown}" if name in out else shown
    return out


def _loggable_header_value(name, value):
    """One header's value as text, with any secret redacted"""
    if name in _SECRET_HEADERS:
        return "<redacted>"
    if name == "set-cookie":
        # browsers join repeated Set-Cookie headers with newlines
        return ", ".join(_cookie_name(x) for x in str(value).splitlines())
    if name == "cookie":
        return "; ".join(_cookie_name(x) for x in str(value).split(";"))
    return str(value)


def _cookie_name(cookie):
    """A cookie with its value (and attributes) redacted"""
    return cookie.split("=", 1)[0].strip() + "=<redacted>"


def body_preview(raw, content_type="", limit=2000):
    """Start of a response body as text, if it is text at all"""
    if not raw or b"%PDF-" in raw[:1024]:
        return None
    textual = not content_type or any(
        kind in content_type for kind in ("text", "json", "xml", "html")
    )
    if not textual:
        return None
    return raw[:limit].decode("utf-8", errors="replace")


def document_summary(doc):
    """Describe a document compactly for an event

    A loading backend may convert a PDF to a single block of text and
    note the original format and page count in ``doc.attrs``
    (``doc_type``, ``num_pages``); those are used when present.
    """
    pages = getattr(doc, "pages", None) or []
    text = getattr(doc, "text", "") or ""
    doc_type = str(doc.attrs.get("doc_type") or "").casefold()
    return {
        "source": doc.attrs.get("source"),
        "pdf": type(doc).__name__ == "PDFDocument" or doc_type == "pdf",
        "pages": doc.attrs.get("num_pages") or len(pages),
        "characters": len(text),
    }


class ResponseRecorder:
    """Usage tracker stand-in that remembers the API response metadata

    Usage updates are forwarded to the wrapped tracker, if there is one,
    so token accounting is unchanged.
    """

    def __init__(self, tracker=None):
        self.tracker = tracker
        self.metadata = None

    def update_from_model(
        self, model=None, response=None, sub_label="default"
    ):
        """Remember the response details, then forward the update"""
        self.metadata = response_metadata(model, response)
        if self.tracker is not None:
            self.tracker.update_from_model(model, response, sub_label)


def response_metadata(model, response):
    """Summarize an API response to diagnose empty or cut answers"""
    choice = (getattr(response, "choices", None) or [None])[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) or ""
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    tokens = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
    }
    return {
        "model": model,
        "finish_reason": getattr(choice, "finish_reason", None),
        **tokens,
        "content_chars": len(content),
        **token_costs(model, **tokens),
    }


def token_costs(
    model, prompt_tokens=None, completion_tokens=None, reasoning_tokens=None
):
    """Dollar cost of a call's tokens at the configured model rates

    Reasoning tokens are billed as output and are already included in
    the completion tokens, so ``reasoning_cost`` is part of
    ``completion_cost``, not an addition to it. Costs are ``None`` when
    the model has no configured rates.
    """
    if model not in LLM_COST_REGISTRY:
        return dict.fromkeys(
            ("prompt_cost", "completion_cost", "reasoning_cost", "total_cost")
        )
    prompt, completion = prompt_tokens or 0, completion_tokens or 0
    return {
        "prompt_cost": cost_for_model(model, prompt, 0),
        "completion_cost": cost_for_model(model, 0, completion),
        "reasoning_cost": cost_for_model(model, 0, reasoning_tokens or 0),
        "total_cost": cost_for_model(model, prompt, completion),
    }
