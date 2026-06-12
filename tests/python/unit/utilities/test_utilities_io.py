"""COMPASS I/O tests"""

import os
import json
from pathlib import Path

import pytest

from compass.utilities.io import load_config, ConfigType, resolve_all_paths
from compass.services.cpu import FileLoader
from compass.exceptions import COMPASSValueError


PYT_CMD = os.getenv("TESSERACT_CMD")


def test_file_loader_sets_default_omp_num_threads(monkeypatch):
    """Test process pool defaults OMP_NUM_THREADS to 1"""

    class DummyPool:
        def __init__(self, *__, **___):
            pass

        def shutdown(self, wait=True, cancel_futures=True):
            return None

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr("compass.services.cpu.ProcessPoolExecutor", DummyPool)

    service = FileLoader()
    service.acquire_resources()

    assert os.environ["OMP_NUM_THREADS"] == "1"

    service.release_resources()


def test_file_loader_preserves_existing_omp_num_threads(monkeypatch):
    """Test process pool preserves user OMP_NUM_THREADS override"""

    class DummyPool:
        def __init__(self, *__, **___):
            pass

        def shutdown(self, wait=True, cancel_futures=True):
            return None

    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setattr("compass.services.cpu.ProcessPoolExecutor", DummyPool)

    service = FileLoader()
    service.acquire_resources()

    assert os.environ["OMP_NUM_THREADS"] == "4"

    service.release_resources()


def test_resolve_all_paths():
    """Test resolving all paths"""

    base_dir = Path.home()

    assert resolve_all_paths("test", base_dir) == "test"
    assert resolve_all_paths("~test", base_dir) == "~test"
    assert (
        resolve_all_paths("/test/f.csv", base_dir)
        == Path("/test/f.csv").as_posix()
    )
    assert (
        resolve_all_paths("./test", base_dir) == (base_dir / "test").as_posix()
    )
    assert resolve_all_paths("../", base_dir) == base_dir.parent.as_posix()
    assert resolve_all_paths(".././", base_dir) == base_dir.parent.as_posix()
    assert (
        resolve_all_paths("../test_file.json", base_dir)
        == (base_dir.parent / "test_file.json").as_posix()
    )
    assert (
        resolve_all_paths("../test_dir/./../", base_dir)
        == base_dir.parent.as_posix()
    )
    assert (
        resolve_all_paths("test_dir/./", base_dir)
        == Path("test_dir").resolve().as_posix()
    )
    assert (
        resolve_all_paths("test_dir/../", base_dir)
        == Path("test_dir").resolve().parent.as_posix()
    )
    assert (
        resolve_all_paths("~/test_dir/../", base_dir) == Path.home().as_posix()
    )


@pytest.mark.parametrize(
    "input_,expected",
    [
        (r".\test", lambda base_dir: (base_dir / "test").as_posix()),
        (
            r"..\test_file.json",
            lambda base_dir: (base_dir.parent / "test_file.json").as_posix(),
        ),
        (
            r"test_dir\..\\test_file.json",
            lambda _base_dir: (
                Path("test_dir/../test_file.json").resolve().as_posix()
            ),
        ),
    ],
)
def test_resolve_all_paths_windows_style_relative_paths(input_, expected):
    """Test resolving Windows-style relative paths on any host"""

    base_dir = Path.home()
    assert resolve_all_paths(input_, base_dir) == expected(base_dir)


def test_resolve_all_paths_list():
    """Test resolving all paths in a list"""
    base_dir = Path.home()
    input_ = [
        "test",
        "./test",
        "../",
        ".././",
        "../test_file.json",
        "../test_dir/./../",
        ["test", "../test_dir/./../"],
    ]
    expected_output = [
        "test",
        (base_dir / "test").as_posix(),
        base_dir.parent.as_posix(),
        base_dir.parent.as_posix(),
        (base_dir.parent / "test_file.json").as_posix(),
        base_dir.parent.as_posix(),
        ["test", base_dir.parent.as_posix()],
    ]

    assert resolve_all_paths(input_, base_dir) == expected_output


def test_resolve_all_paths_dict():
    """Test resolving all paths in a dict"""
    base_dir = Path.home()
    input_ = {
        "a": "test",
        "b": "./test",
        "c": "../",
        "d": ".././",
        "e": "../test_file.json",
        "f": "../test_dir/./../",
        "g": ["test", "../test_dir/./../"],
        "h": {
            "a": "test",
            "b": ["test", "../test_dir/./../"],
        },
    }
    expected_output = {
        "a": "test",
        "b": (base_dir / "test").as_posix(),
        "c": base_dir.parent.as_posix(),
        "d": base_dir.parent.as_posix(),
        "e": (base_dir.parent / "test_file.json").as_posix(),
        "f": base_dir.parent.as_posix(),
        "g": [
            "test",
            base_dir.parent.as_posix(),
        ],
        "h": {
            "a": "test",
            "b": [
                "test",
                base_dir.parent.as_posix(),
            ],
        },
    }

    assert resolve_all_paths(input_, base_dir) == expected_output


@pytest.mark.parametrize("config_type", list(ConfigType))
def test_write_load_config(tmp_path, config_type):
    """Test loading a configuration file"""

    base_fn = f"test.{config_type}"

    test_dictionary = {"a": 1, "b": 2}
    with Path(tmp_path / base_fn).open("w", encoding="utf-8") as config_file:
        config_type.dump(test_dictionary, config_file)

    assert load_config(tmp_path / "." / base_fn) == test_dictionary

    test_dictionary = {
        "a": 1,
        "b": "A string",
        "path_a": "./config.json",
        "path_b": "./../another.json",
        "path_c": "./something/.././../another.json",
    }
    config_type.write(tmp_path / base_fn, test_dictionary)

    expected_dict = {
        "a": 1,
        "b": "A string",
        "path_a": (tmp_path / "config.json").as_posix(),
        "path_b": (tmp_path.parent / "another.json").as_posix(),
        "path_c": (tmp_path.parent / "another.json").as_posix(),
    }
    assert load_config(tmp_path / "." / base_fn) == expected_dict

    assert (
        load_config(tmp_path / "." / base_fn, resolve_paths=False)
        == test_dictionary
    )


@pytest.mark.parametrize("config_type", list(ConfigType))
def test_config_dumps_loads(config_type):
    """Test dumping and loading a configuration file to and from a str"""

    test_dictionary = {
        "a": 1,
        "b": "A string",
        "path_a": "./config.json",
        "path_b": "./../another.json",
        "path_c": "./something/.././../another.json",
    }
    assert (
        config_type.loads(config_type.dumps(test_dictionary))
        == test_dictionary
    )


def test_load_config_json(tmp_path):
    """Test `load_config` with JSON file"""

    config_data = {"key": "value", "number": 42}
    config_file = tmp_path / "test_config.json"
    with config_file.open("w", encoding="utf-8") as f:
        json.dump(config_data, f)

    result = load_config(config_file)
    assert result == config_data


def test_load_config_json5(tmp_path):
    """Test `load_config` with JSON5 file"""

    config_content = """{
        // This is a comment
        "key": "value",
        "number": 42,
    }"""
    config_file = tmp_path / "test_config.json5"
    with config_file.open("w", encoding="utf-8") as f:
        f.write(config_content)

    result = load_config(config_file)
    assert result == {"key": "value", "number": 42}


def test_load_config_invalid_extension(tmp_path):
    """Test `load_config` with invalid file extension"""

    config_file = tmp_path / "test_config.txt"
    config_file.touch()

    with pytest.raises(
        COMPASSValueError,
        match=(
            r"Got unknown config file extension: '.txt'. Supported "
            r"extensions are:"
        ),
    ):
        load_config(config_file)


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
