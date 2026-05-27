"""Tests for compass._cli.process"""

from pathlib import Path

import pytest

from click import ClickException

import compass._cli.process as process_module
from compass._cli.process import (
    _next_versioned_directory,
    _resolve_out_dir_conflict,
)


def test_next_versioned_directory_skips_existing_versions(tmp_path):
    """Find the next available versioned output directory"""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (tmp_path / "outputs_v2").mkdir()

    result = _next_versioned_directory(out_dir)

    assert result == tmp_path / "outputs_v3"


def test_resolve_out_dir_conflict_increment(tmp_path):
    """Increment output directory when policy is increment"""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()

    result = _resolve_out_dir_conflict(out_dir, "increment")

    assert result == tmp_path / "outputs_v2"


def test_resolve_out_dir_conflict_overwrite(tmp_path):
    """Remove existing directory when policy is overwrite"""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "temp.txt").write_text("x", encoding="utf-8")

    result = _resolve_out_dir_conflict(out_dir, "overwrite")

    assert result == out_dir
    assert not out_dir.exists()


def test_resolve_out_dir_conflict_prompt_increment(tmp_path, monkeypatch):
    """Prompt mode can select incremented directory"""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()

    monkeypatch.setattr(process_module.sys, "stdin", _Tty())
    monkeypatch.setattr(process_module.click, "confirm", lambda *_, **__: True)

    result = _resolve_out_dir_conflict(out_dir, "prompt")

    assert result == tmp_path / "outputs_v2"


def test_resolve_out_dir_conflict_prompt_overwrite(tmp_path, monkeypatch):
    """Prompt mode can select overwrite directory"""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "temp.txt").write_text("x", encoding="utf-8")

    monkeypatch.setattr(process_module.sys, "stdin", _Tty())
    answers = iter([False, True])
    monkeypatch.setattr(
        process_module.click,
        "confirm",
        lambda *_, **__: next(answers),
    )

    result = _resolve_out_dir_conflict(out_dir, "prompt")

    assert result == out_dir
    assert not out_dir.exists()


def test_resolve_out_dir_conflict_prompt_cancel(tmp_path, monkeypatch):
    """Prompt mode raises if user declines both options"""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()

    monkeypatch.setattr(process_module.sys, "stdin", _Tty())
    answers = iter([False, False])
    monkeypatch.setattr(
        process_module.click,
        "confirm",
        lambda *_, **__: next(answers),
    )

    with pytest.raises(ClickException, match="Run cancelled"):
        _ = _resolve_out_dir_conflict(out_dir, "prompt")


class _NoTty:
    def isatty(self):
        return False


class _Tty:
    def isatty(self):
        return True


def test_resolve_out_dir_conflict_prompt_non_interactive(
    tmp_path, monkeypatch
):
    """Prompt mode raises in non-interactive mode"""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()

    monkeypatch.setattr(process_module.sys, "stdin", _NoTty())

    with pytest.raises(ClickException, match="non-interactive"):
        _ = _resolve_out_dir_conflict(out_dir, "prompt")


def test_resolve_out_dir_conflict_fail_keeps_path(tmp_path):
    """Fail policy leaves existing output directory unchanged"""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()

    result = _resolve_out_dir_conflict(out_dir, "fail")

    assert result == out_dir
    assert out_dir.exists()


def test_process_uses_prompt_policy_in_interactive_terminal(
    tmp_path, monkeypatch
):
    """Auto-select prompt policy when stdin is a TTY"""
    monkeypatch.setattr(process_module.sys, "stdin", _Tty())

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()

    confirmed = []
    monkeypatch.setattr(
        process_module.click,
        "confirm",
        lambda *_, **__: confirmed.append(True) or True,
    )

    result = (
        process_module._resolve_out_dir_conflict.__wrapped__
        if hasattr(process_module._resolve_out_dir_conflict, "__wrapped__")
        else None
    )

    policy = "prompt" if process_module.sys.stdin.isatty() else "fail"
    assert policy == "prompt"


def test_process_uses_fail_policy_in_non_interactive_terminal(monkeypatch):
    """Auto-select fail policy when stdin is not a TTY"""
    monkeypatch.setattr(process_module.sys, "stdin", _NoTty())

    policy = "prompt" if process_module.sys.stdin.isatty() else "fail"
    assert policy == "fail"


def test_process_flag_overrides_tty_detection(tmp_path, monkeypatch):
    """Explicit --out_dir_exists flag overrides auto-TTY detection"""
    monkeypatch.setattr(process_module.sys, "stdin", _Tty())

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()

    explicit_flag = "increment"
    policy = (
        explicit_flag
        if explicit_flag
        else ("prompt" if process_module.sys.stdin.isatty() else "fail")
    )
    result = _resolve_out_dir_conflict(out_dir, policy)
    assert result == tmp_path / "outputs_v2"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
