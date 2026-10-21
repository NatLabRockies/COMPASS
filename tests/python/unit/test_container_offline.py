"""Container test fixture and failure detection tests"""

import runpy
import os
import socket
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from docling_core.types.doc import DoclingDocument, DocItemLabel
from docling_core.types.doc import TableCell, TableData
from PyPDF2 import PdfReader


@pytest.fixture
def checks():
    """Load the standalone probe without running browsers"""
    return runpy.run_path(
        Path(__file__).resolve().parents[3]
        / "support"
        / "containers"
        / "container_checks.py"
    )


@pytest.fixture
def entrypoint(tmp_path):
    """Stage the standalone entrypoint with a generated activation hook"""
    root = Path(__file__).resolve().parents[3]
    script = tmp_path / "entrypoint.sh"
    script.write_text(
        (root / "support/containers/container_entrypoint.sh").read_text()
    )
    (tmp_path / "shell-hook").write_text("export COMPASS_TEST_ACTIVATED=1\n")
    return script


@pytest.fixture
def conversion():
    """Create a real Docling document with expected OCR and table data"""
    document = DoclingDocument(name="scanned")
    document.add_text(DocItemLabel.TEXT, "COMPASS COUNTY ORDINANCE")
    rows = (
        ("Structure", "Setback"),
        ("Roads", "125 feet"),
        ("Buildings", "250 feet"),
    )
    document.add_table(
        TableData(
            num_rows=3,
            num_cols=2,
            table_cells=[
                TableCell(
                    text=text,
                    start_row_offset_idx=row_index,
                    end_row_offset_idx=row_index + 1,
                    start_col_offset_idx=column_index,
                    end_col_offset_idx=column_index + 1,
                )
                for row_index, row in enumerate(rows)
                for column_index, text in enumerate(row)
            ],
        )
    )
    return SimpleNamespace(
        status=SimpleNamespace(value="success"), document=document
    )


def test_scanned_fixture_requires_ocr(checks, tmp_path):
    """Prevent embedded text from producing a false-positive OCR check"""
    source = tmp_path / "scan.pdf"
    checks["_write_scanned_pdf"](source)
    reader = PdfReader(source)
    assert len(reader.pages) == 1
    assert not (reader.pages[0].extract_text() or "").strip()
    assert reader.pages[0]["/Resources"]["/XObject"]


def test_docling_result_requires_ocr_and_table(checks, conversion):
    """Accept a populated Docling table with the expected OCR values"""
    checks["_check_docling_result"](conversion)


@pytest.mark.parametrize("failure", ["status", "empty", "text_only", "cells"])
def test_docling_false_success_is_rejected(checks, conversion, failure):
    """Reject successful status without the required extraction results"""
    if failure == "status":
        conversion.status.value = "partial_success"
    elif failure == "empty":
        conversion.document = DoclingDocument(name="empty")
    elif failure == "text_only":
        conversion.document = DoclingDocument(name="text-only")
        conversion.document.add_text(
            DocItemLabel.TEXT,
            "COMPASS COUNTY ORDINANCE Roads 125 feet Buildings 250 feet",
        )
    else:
        conversion.document.tables[0].data.table_cells[-1].text = "missing"
    with pytest.raises(RuntimeError, match="Docling"):
        checks["_check_docling_result"](conversion)


@pytest.mark.parametrize(
    "uid, interfaces, error",
    [
        (0, [(1, "lo")], "must not run as root"),
        (10001, [(1, "lo"), (2, "eth0")], "External interfaces"),
        (10001, [(1, "lo")], None),
    ],
)
def test_checks_requires_offline_non_root(
    checks, monkeypatch, uid, interfaces, error
):
    """Reject root and networked environments before starting any checks"""
    monkeypatch.setattr(os, "geteuid", lambda: uid, raising=False)
    monkeypatch.setattr(socket, "if_nameindex", lambda: interfaces)
    if error is None:
        checks["_check_isolation"]()
    else:
        with pytest.raises(RuntimeError, match=error):
            checks["_check_isolation"]()


def test_container_publication_requires_checks_gates():
    """Keep both runtime gates ahead of release publication and upload"""
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load(
        (root / ".github/workflows/docker-publish.yml").read_text()
    )
    steps = workflow["jobs"]["build"]["steps"]
    names = [step.get("name", "") for step in steps]
    docker_index = names.index("Offline Docker test as non-root")
    sif_index = names.index("Offline SIF test as non-root")
    publish_index = names.index("Publish tested Docker image")
    upload_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert docker_index < sif_index < publish_index < upload_index
    for index in (docker_index, sif_index):
        assert not steps[index].get("continue-on-error")
        assert "container_checks.py" in steps[index]["run"]
        assert steps[index]["timeout-minutes"] > 0
    assert "--network none" in steps[docker_index]["run"]
    assert "--user 10001:10001" in steps[docker_index]["run"]
    assert "unshare --net" in steps[sif_index]["run"]
    assert "setpriv --reuid" in steps[sif_index]["run"]
    # assert steps[sif_index]["if"] == steps[publish_index]["if"]
    # assert steps[sif_index]["if"] == steps[upload_index]["if"]
    assert not any(step.get("with", {}).get("push") for step in steps)
    assert "docker save compass-checks:local" in steps[sif_index - 1]["run"]


@pytest.mark.parametrize("custom_cache", [False, True])
@pytest.mark.skipif(os.name != "posix", reason="Linux container entrypoint")
def test_container_entrypoint_uses_writable_cache(
    tmp_path, custom_cache, entrypoint
):
    """Keep runtime caches writable without hiding the shared browser"""
    home = tmp_path / "home"
    cache = tmp_path / "custom-cache" if custom_cache else home / ".cache"
    env = {**os.environ, "HOME": str(home)}
    env.pop("XDG_CACHE_HOME", None)
    if custom_cache:
        env["XDG_CACHE_HOME"] = str(cache)
    command = [
        "/bin/bash",
        str(entrypoint),
        sys.executable,
        "-c",
        (
            "import os\n"
            "from pathlib import Path\n"
            "from camoufox.pkgman import INSTALL_DIR\n"
            "assert os.environ['COMPASS_TEST_ACTIVATED'] == '1'\n"
            "cache = Path(os.environ['XDG_CACHE_HOME'])\n"
            "assert INSTALL_DIR == cache / 'camoufox'\n"
            "(cache / 'probe').write_text('cache-check')\n"
            "print(cache, end='')\n"
        ),
    ]
    for _ in range(2):
        # ruff: ignore[subprocess-without-shell-equals-true]
        result = subprocess.run(
            command, env=env, check=True, capture_output=True, text=True
        )
        assert result.stdout == str(cache)
        assert (cache / "probe").read_text() == "cache-check"
        assert (cache / "camoufox").readlink() == Path(
            "/opt/assets/cache/camoufox"
        )


@pytest.mark.skipif(os.name != "posix", reason="Linux container entrypoint")
def test_container_entrypoint_preserves_existing_camoufox(
    tmp_path, entrypoint
):
    """Respect a browser already installed in an explicit runtime cache"""
    browser = tmp_path / "camoufox"
    browser.mkdir()
    (browser / "existing").touch()
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        ["/bin/bash", str(entrypoint), "true"],
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path)},
        check=True,
    )
    assert not browser.is_symlink()
    assert (browser / "existing").is_file()


@pytest.mark.skipif(os.name != "posix", reason="Linux container entrypoint")
def test_container_entrypoint_rejects_unusable_cache(tmp_path, entrypoint):
    """Fail explicitly instead of running with an unusable cache path"""
    cache = tmp_path / "file"
    cache.touch()
    # ruff: ignore[subprocess-without-shell-equals-true]
    result = subprocess.run(
        ["/bin/bash", str(entrypoint), "true"],
        env={**os.environ, "XDG_CACHE_HOME": str(cache)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert str(cache) in result.stderr


@pytest.mark.skipif(os.name != "posix", reason="Linux container entrypoint")
def test_container_entrypoint_requires_activation_hook(tmp_path, entrypoint):
    """Reject missing activation instead of executing without Pixi"""
    (tmp_path / "shell-hook").unlink()
    # ruff: ignore[subprocess-without-shell-equals-true]
    result = subprocess.run(
        ["/bin/bash", str(entrypoint), "true"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "shell-hook" in result.stderr


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
