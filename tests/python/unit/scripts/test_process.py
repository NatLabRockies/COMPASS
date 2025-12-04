"""Tests for compass.scripts.process"""

import logging
from pathlib import Path
from itertools import product

import pytest

from compass.exceptions import COMPASSValueError
import compass.scripts.process as process_module
from compass.scripts.process import (
    _COMPASSRunner,
    process_jurisdictions_with_openai,
)
from compass.utilities import ProcessKwargs


@pytest.fixture
def testing_log_file(tmp_path):
    """Logger fixture for testing"""
    log_fp = tmp_path / "test.log"
    handler = logging.FileHandler(log_fp, encoding="utf-8")
    logger = logging.getLogger("compass")
    prev_level = logger.level
    prev_propagate = logger.propagate
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    logger.addHandler(handler)

    yield log_fp

    handler.flush()
    logger.removeHandler(handler)
    handler.close()
    logger.setLevel(prev_level)
    logger.propagate = prev_propagate


@pytest.fixture
def patched_runner(monkeypatch):
    """Patch the COMPASSRunner to a dummy that bypasses processing"""

    class DummyRunner:
        """Minimal runner that bypasses full processing"""

        def __init__(self, **_):
            pass

        async def run(self, jurisdiction_fp):
            return f"processed {jurisdiction_fp}"

    monkeypatch.setattr(process_module, "_COMPASSRunner", DummyRunner)


def test_known_local_docs_missing_file(tmp_path):
    """Raise when known_local_docs points to missing config"""
    missing_fp = tmp_path / "does_not_exist.json"
    runner = _COMPASSRunner(
        dirs=None,
        log_listener=None,
        tech="solar",
        models={},
        process_kwargs=ProcessKwargs(str(missing_fp), None),
    )

    with pytest.raises(COMPASSValueError, match="Config file does not exist"):
        _ = runner.known_local_docs


def test_known_local_docs_logs_missing_file(tmp_path, testing_log_file):
    """Log missing known_local_docs config to error file"""

    missing_fp = tmp_path / "does_not_exist.json"
    runner = _COMPASSRunner(
        dirs=None,
        log_listener=None,
        tech="solar",
        models={},
        process_kwargs=ProcessKwargs(str(missing_fp), None),
    )

    with pytest.raises(COMPASSValueError, match="Config file does not exist"):
        _ = runner.known_local_docs

    assert testing_log_file.exists()
    assert "Config file does not exist" in testing_log_file.read_text(
        encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_duplicate_tasks_logs_to_file(tmp_path):
    """Log duplicate LLM tasks to error file"""

    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.touch()

    with pytest.raises(COMPASSValueError, match="Found duplicated task"):
        _ = await process_jurisdictions_with_openai(
            out_dir=tmp_path / "outputs",
            tech="solar",
            jurisdiction_fp=jurisdiction_fp,
            model=[
                {
                    "name": "gpt-4.1-mini",
                    "tasks": ["default", "date_extraction"],
                },
                {
                    "name": "gpt-4.1",
                    "tasks": [
                        "ordinance_text_extraction",
                        "permitted_use_text_extraction",
                        "date_extraction",
                    ],
                },
            ],
        )

    log_files = list((tmp_path / "outputs" / "logs").glob("*"))
    assert len(log_files) == 1
    assert "Fatal error during processing" not in log_files[0].read_text(
        encoding="utf-8"
    )
    assert "Found duplicated task" in log_files[0].read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_external_exceptions_logged_to_file(tmp_path, monkeypatch):
    """Log external exceptions to error file"""

    def _always_fail(*__, **___):
        raise NotImplementedError("Simulated external error")

    monkeypatch.setattr(
        process_module, "_initialize_model_params", _always_fail
    )

    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.touch()

    with pytest.raises(NotImplementedError, match="Simulated external error"):
        _ = await process_jurisdictions_with_openai(
            out_dir=tmp_path / "outputs",
            tech="solar",
            jurisdiction_fp=jurisdiction_fp,
        )

    log_files = list((tmp_path / "outputs" / "logs").glob("*"))
    assert len(log_files) == 1
    assert "Fatal error during processing" in log_files[0].read_text(
        encoding="utf-8"
    )
    assert "Simulated external error" in log_files[0].read_text(
        encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_process_args_logged_at_debug_to_file(
    tmp_path, patched_runner, assert_message_was_logged
):
    """Log function arguments with DEBUG_TO_FILE level"""

    out_dir = tmp_path / "outputs"
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.touch()

    result = await process_jurisdictions_with_openai(
        out_dir=out_dir,
        tech="solar",
        jurisdiction_fp=jurisdiction_fp,
        log_level="DEBUG",
    )

    assert result == f"processed {jurisdiction_fp}"

    assert_message_was_logged(
        "Called 'process_jurisdictions_with_openai' with:",
        log_level="DEBUG_TO_FILE",
    )
    assert_message_was_logged('"out_dir": ', log_level="DEBUG_TO_FILE")
    assert_message_was_logged("outputs", log_level="DEBUG_TO_FILE")
    assert_message_was_logged('"tech": "solar"', log_level="DEBUG_TO_FILE")
    assert_message_was_logged('"jurisdiction_fp": ', log_level="DEBUG_TO_FILE")
    assert_message_was_logged("jurisdictions.csv", log_level="DEBUG_TO_FILE")
    assert_message_was_logged(
        '"log_level": "DEBUG"', log_level="DEBUG_TO_FILE"
    )
    assert_message_was_logged(
        '"model": "gpt-4o-mini"', log_level="DEBUG_TO_FILE"
    )
    assert_message_was_logged(
        '"keep_async_logs": false', log_level="DEBUG_TO_FILE"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "has_known_local_docs",
        "has_known_doc_urls",
        "perform_se_search",
        "perform_website_search",
    ),
    [
        pytest.param(
            *flags, id=("local-{}_urls-{}_se-{}_web-{}".format(*flags))
        )
        for flags in product([False, True], repeat=4)
    ],
)
async def test_process_steps_logged(
    tmp_path,
    patched_runner,
    assert_message_was_logged,
    has_known_local_docs,
    has_known_doc_urls,
    perform_se_search,
    perform_website_search,
):
    """Log enabled processing steps for every combination of inputs"""

    out_dir = tmp_path / "outputs"
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.touch()

    known_local_docs = None
    if has_known_local_docs:
        known_local_docs = {"1": [{"source_fp": tmp_path / "local_doc.pdf"}]}

    known_doc_urls = None
    if has_known_doc_urls:
        known_doc_urls = {
            "1": [{"source": "https://example.com/ordinance.pdf"}]
        }

    expected_steps = []
    if has_known_local_docs:
        expected_steps.append("Check local document")
    if has_known_doc_urls:
        expected_steps.append("Check known document URL")
    if perform_se_search:
        expected_steps.append("Look for document using search engine")
    if perform_website_search:
        expected_steps.append("Look for document on jurisdiction website")

    if not expected_steps:
        with pytest.raises(
            COMPASSValueError, match="No processing steps enabled"
        ):
            await process_jurisdictions_with_openai(
                out_dir=str(out_dir),
                tech="solar",
                jurisdiction_fp=str(jurisdiction_fp),
                log_level="DEBUG",
                known_local_docs=known_local_docs,
                known_doc_urls=known_doc_urls,
                perform_se_search=perform_se_search,
                perform_website_search=perform_website_search,
            )
        return

    result = await process_jurisdictions_with_openai(
        out_dir=str(out_dir),
        tech="solar",
        jurisdiction_fp=str(jurisdiction_fp),
        log_level="DEBUG",
        known_local_docs=known_local_docs,
        known_doc_urls=known_doc_urls,
        perform_se_search=perform_se_search,
        perform_website_search=perform_website_search,
    )

    assert result == f"processed {jurisdiction_fp}"

    assert_message_was_logged(
        "Using the following processing step(s):", log_level="INFO"
    )
    assert_message_was_logged(" -> ".join(expected_steps), log_level="INFO")


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
