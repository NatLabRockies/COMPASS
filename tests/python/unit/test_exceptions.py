"""Test COMPASS exception types

Most exception logic + tests pulled from GAPs
(https://github.com/NREL/gaps)
"""

from pathlib import Path

import pytest

from compass.exceptions import (
    COMPASSError,
    COMPASSValueError,
    COMPASSNotInitializedError,
    COMPASSRuntimeError,
)


BASIC_ERROR_MESSAGE = "An error message"


def test_exceptions_log_error(caplog, assert_message_was_logged):
    """Test that a raised exception logs message, if any."""

    try:
        raise COMPASSError
    except COMPASSError:
        pass

    assert not caplog.records

    try:
        raise COMPASSError(BASIC_ERROR_MESSAGE)
    except COMPASSError:
        pass

    assert_message_was_logged("COMPASSError", "ERROR")
    assert_message_was_logged(BASIC_ERROR_MESSAGE, "ERROR")


def test_exceptions_log_uncaught_error(assert_message_was_logged):
    """Test that a raised exception logs message if uncaught."""

    with pytest.raises(COMPASSError):
        raise COMPASSError(BASIC_ERROR_MESSAGE)

    assert_message_was_logged("COMPASSError", "ERROR")
    assert_message_was_logged(BASIC_ERROR_MESSAGE, "ERROR")


@pytest.mark.parametrize(
    "raise_type, catch_types",
    [
        (
            COMPASSNotInitializedError,
            [COMPASSError, COMPASSNotInitializedError],
        ),
        (
            COMPASSValueError,
            [COMPASSError, ValueError, COMPASSValueError],
        ),
        (
            COMPASSRuntimeError,
            [COMPASSError, RuntimeError, COMPASSRuntimeError],
        ),
    ],
)
def test_catching_error_by_type(
    raise_type, catch_types, assert_message_was_logged
):
    """Test that gaps exceptions are caught correctly."""
    for catch_type in catch_types:
        with pytest.raises(catch_type) as exc_info:
            raise raise_type(BASIC_ERROR_MESSAGE)

        assert BASIC_ERROR_MESSAGE in str(exc_info.value)
        assert_message_was_logged(raise_type.__name__, "ERROR")
        assert_message_was_logged(BASIC_ERROR_MESSAGE, "ERROR")


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
