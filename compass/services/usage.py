"""Ordinances usage tracking utilities"""

import time
import logging
from collections import UserDict, deque
from functools import total_ordering

from compass.exceptions import COMPASSValueError


logger = logging.getLogger(__name__)
LLM_USAGE_RATES_KEY = "_llm_usage_rates"


@total_ordering
class TimedEntry:
    """An entry that performs comparisons based on time added, not value

    Examples
    --------
    >>> a = TimedEntry(100)
    >>> a > 1000
    True
    """

    def __init__(self, value):
        """

        Parameters
        ----------
        value : object
            Some value to store as an entry.
        """
        self._value = value
        self._time = time.perf_counter()

    @property
    def time(self):
        """float: Time the entry was added to the tracker"""
        return self._time

    @property
    def value(self):
        """object: Value that was added to the tracker"""
        return self._value

    def __eq__(self, other):
        return self.time == other

    def __lt__(self, other):
        return self.time < other

    def __hash__(self):
        return hash((self.value, self.time))


class TimeBoundedUsageTracker:
    """Track usage of a resource over time

    This class wraps a double-ended queue, and any inputs older than
    a certain time are dropped. Those values are also subtracted from
    the running total.

    References
    ----------
    https://stackoverflow.com/questions/51485656/efficient-time-bound-queue-in-python
    """

    def __init__(self, max_seconds=70):
        """

        Parameters
        ----------
        max_seconds : int, optional
            Maximum age in seconds of an element before it is dropped
            from consideration. By default, ``65``.
        """
        self.max_seconds = max_seconds
        self._total = 0
        self._q = deque()

    @property
    def total(self):
        """float: Total of all entries younger than `max_seconds`"""
        self._discard_old_values()
        return self._total

    def add(self, value):
        """Add a value to track

        Parameters
        ----------
        value : int or float
            A new value to add to the queue. It's total will be added to
            the running total, and it will live for `max_seconds` before
            being discarded.

        Returns
        -------
        float
            Timestamp stored with the value.
        """
        entry = TimedEntry(value)
        self._q.append(entry)
        self._total += value
        return entry.time

    def _discard_old_values(self):
        """Discard 'old' values from the queue"""
        cutoff_time = time.perf_counter() - self.max_seconds
        try:
            while self._q[0] < cutoff_time:
                self._total -= self._q.popleft().value
        except IndexError:
            pass


class _OnlineUsageSummary:
    """Track online minimum, mean, and maximum values"""

    def __init__(self, count=0, total=0, minimum=None, maximum=None):
        self._count = count
        self._total = total
        self._minimum = minimum
        self._maximum = maximum

    def add(self, value):
        """Add one value to the summary"""
        self._count += 1
        self._total += value
        self._minimum = (
            value if self._minimum is None else min(self._minimum, value)
        )
        self._maximum = (
            value if self._maximum is None else max(self._maximum, value)
        )

    def copy(self):
        """Return an independent copy of this summary"""
        return self.__class__(
            self._count, self._total, self._minimum, self._maximum
        )

    def as_dict(self):
        """dict: Serialized minimum, mean, and maximum values"""
        if self._count == 0:
            return {"min": 0, "mean": 0, "max": 0}

        return {
            "min": self._minimum,
            "mean": self._total / self._count,
            "max": self._maximum,
        }


class _FixedWindowUsageTracker:
    """Track values in fixed windows with constant-size state"""

    def __init__(self, window_seconds, start_time):
        self.window_seconds = window_seconds
        self.start_time = start_time
        self._bucket_index = 0
        self._current_value = 0
        self._summary = _OnlineUsageSummary()

    def add(self, value, timestamp=None):
        """Add a value at a monotonic timestamp"""
        if timestamp is None:
            timestamp = time.perf_counter()

        bucket_index = int(
            (timestamp - self.start_time) // self.window_seconds
        )
        if bucket_index < self._bucket_index:
            msg = "Usage timestamps must be monotonically increasing"
            raise COMPASSValueError(msg)

        if bucket_index == self._bucket_index:
            self._current_value += value
            return

        self._summary.add(self._current_value)
        self._bucket_index = bucket_index
        self._current_value = value

    def snapshot(self):
        """dict: Summary including the current partial time window"""
        timestamp = time.perf_counter()

        bucket_index = int(
            (timestamp - self.start_time) // self.window_seconds
        )
        bucket_index = max(bucket_index, self._bucket_index)
        summary = self._summary.copy()
        summary.add(self._current_value)
        return summary.as_dict()


class _ConcurrentRequestsTracker:
    """Track the number of concurrent requests"""

    def __init__(self):
        self._active_requests = 0
        self._summary = _OnlineUsageSummary()

    def start_request(self):
        """Start tracking a new concurrent request"""
        self._active_requests += 1
        self._summary.add(self._active_requests)

    def end_request(self):
        """Stop tracking an active concurrent request"""
        self._active_requests = max(0, self._active_requests - 1)

    def snapshot(self):
        """dict: Snapshot of concurrent requests summary"""
        summary = self._summary.copy()
        summary.add(self._active_requests)
        return summary.as_dict()


class _ModelUsageRateTracker:
    """Track fixed-window request and token rates for one scope"""

    def __init__(self, start_time):
        self.requests_per_second = _FixedWindowUsageTracker(1, start_time)
        self.requests_per_minute = _FixedWindowUsageTracker(60, start_time)
        self.tokens_per_minute = _FixedWindowUsageTracker(60, start_time)
        self.concurrent_requests = _ConcurrentRequestsTracker()

    def record_request(self, timestamp):
        """Record a submitted request"""
        self.requests_per_second.add(1, timestamp)
        self.requests_per_minute.add(1, timestamp)

    def record_tokens(self, tokens, timestamp):
        """Record tokens returned by a completed request"""
        self.tokens_per_minute.add(tokens, timestamp)

    def start_request_attempt(self):
        """Record the concurrency when a request attempt starts"""
        self.concurrent_requests.start_request()

    def end_request_attempt(self):
        """Record that an active request attempt ended"""
        self.concurrent_requests.end_request()

    def snapshot(self):
        """dict: Serialized rate summaries"""
        return {
            "requests_per_second": self.requests_per_second.snapshot(),
            "requests_per_minute": self.requests_per_minute.snapshot(),
            "tokens_per_minute": self.tokens_per_minute.snapshot(),
            "concurrent_requests": self.concurrent_requests.snapshot(),
        }


class LLMRateTracker(UserDict):
    """Track run-wide and per-model LLM usage rates on calls"""

    def __init__(self, label=LLM_USAGE_RATES_KEY):
        """

        Parameters
        ----------
        label : str, optional
            Top-level label to use when persisting rate statistics.
            By default, ``"_llm_usage_rates"``.
        """
        super().__init__()
        self.label = label
        self._start_time = time.perf_counter()
        self._overall = _ModelUsageRateTracker(self._start_time)
        self._models = {}

    def record_request(self, model, timestamp):
        """Record a submitted LLM request"""
        self._overall.record_request(timestamp)
        self._model_tracker(model).record_request(timestamp)
        return timestamp

    def record_tokens(self, model, tokens, timestamp):
        """Record actual tokens from a completed LLM request"""
        self._overall.record_tokens(tokens, timestamp)
        self._model_tracker(model).record_tokens(tokens, timestamp)
        return timestamp

    def start_request_attempt(self, model):
        """Record the start of an LLM request attempt"""
        self._overall.start_request_attempt()
        self._model_tracker(model).start_request_attempt()

    def end_request_attempt(self, model):
        """Record the end of an LLM request attempt"""
        self._overall.end_request_attempt()
        self._model_tracker(model).end_request_attempt()

    def _model_tracker(self, model):
        """Return the rate tracker for a model"""
        return self._models.setdefault(
            model, _ModelUsageRateTracker(self._start_time)
        )

    def snapshot(self):
        """dict: Run-wide and per-model rate summaries"""
        self.data = {
            "overall": self._overall.snapshot(),
            "models": {
                model: tracker.snapshot()
                for model, tracker in self._models.items()
            },
        }
        return self

    def add_to(self, other):
        """Add the current rate statistics to another dictionary"""
        other.update({self.label: dict(self.snapshot())})


class LLMUsageTracker(UserDict):
    """Rate or API usage tracker"""

    UNKNOWN_MODEL_LABEL = "unknown_model"
    """Label used in the usage dictionary for unknown models"""

    def __init__(self, label, response_parser):
        """

        Parameters
        ----------
        label : str
            Top-level label to use when adding this usage information to
            another dictionary.
        response_parser : callable
            A callable that takes the current usage info (in dictionary
            format) and an LLm response as inputs, updates the usage
            dictionary with usage info based on the response, and
            returns the updated dictionary. See, for example,
            :func:`compass.services.openai.usage_from_response`.
        """
        super().__init__()
        self.label = label
        self.response_parser = response_parser

    def add_to(self, other):
        """Add the contents of this usage information to another dict

        The contents of this dictionary are stored under the `label`
        key that this object was initialized with.

        Parameters
        ----------
        other : dict
            A dictionary to add the contents of this one to.
        """
        other.update({self.label: {**self, "tracker_totals": self.totals}})

    @property
    def totals(self):
        """dict: Aggregated usage totals across all sub-labels"""
        totals = {}
        for model, model_usage in self.items():
            total_model_usage = totals[model] = {}
            for report in model_usage.values():
                try:
                    sub_label_report = report.items()
                except AttributeError:
                    continue

                for tracked_value, count in sub_label_report:
                    total_model_usage[tracked_value] = (
                        total_model_usage.get(tracked_value, 0) + count
                    )
        return totals

    def update_from_model(
        self, model=None, response=None, sub_label="default"
    ):
        """Update usage from a model response

        Parameters
        ----------
        model : str, optional
            Name of model that usage is being recorded for. If ``None``
            or empty string, the usage will be placed under the
            :obj:`LLMUsageTracker.UNKNOWN_MODEL_LABEL` label.
        response : object, optional
            Model call response, which either contains usage information
            or can be used to infer/compute usage. If ``None``, no
            update is made. By default, ``None``.
        sub_label : str, optional
            Optional label to categorize usage under. This can be used
            to track usage related to certain categories.
            By default, ``"default"``.
        """
        if response is None:
            return

        model_usage = self.setdefault(model or self.UNKNOWN_MODEL_LABEL, {})
        model_usage[sub_label] = self.response_parser(
            model_usage.get(sub_label, {}), response
        )
