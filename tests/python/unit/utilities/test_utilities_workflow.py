"""Tests for the workflow map that event logs link to"""

import ast
import importlib
from pathlib import Path

import pytest

import compass
from compass.utilities.enums import COMPASSDocumentCollectionStep
from compass.utilities.workflow import (
    load_workflow,
    node_outcomes,
    workflow_nodes,
    workflow_problems,
)


PACKAGE_DIR = Path(compass.__file__).parent
REPO_DIR = PACKAGE_DIR.parent


def _source_calls(name):
    """Every call to `name` in the package, as (path, call node)"""
    for path in PACKAGE_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = getattr(func, "id", None) or getattr(func, "attr", None)
            if called == name:
                yield path, node


def _literal_args(call, count):
    """The first `count` positional args, if all are string literals"""
    args = call.args[:count]
    if len(args) < count or not all(
        isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        for arg in args
    ):
        return None
    return [arg.value for arg in args]


def _step_id(path, arg):
    """A step ID argument: a string literal or a module constant"""
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.Name):
        module = ".".join(path.relative_to(REPO_DIR).with_suffix("").parts)
        value = getattr(importlib.import_module(module), arg.id, None)
        return value if isinstance(value, str) else None
    return None


def _defined_names(path):
    """Top-level functions and classes, and methods as Class.method"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for item in tree.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(item.name)
        elif isinstance(item, ast.ClassDef):
            names.add(item.name)
            names.update(
                f"{item.name}.{member.name}"
                for member in item.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
    return names


def test_workflow_map_is_consistent():
    """IDs are unique, parents are groups, and edges lead somewhere"""
    assert workflow_problems() == []


def test_workflow_problems_are_reported():
    """A broken map is described, not silently accepted"""
    broken = {
        "nodes": [
            {"id": "a", "kind": "step", "label": "A"},
            {"id": "a", "kind": "step", "label": "Again"},
            {"id": "b", "kind": "step", "label": "B", "next": "missing"},
            {"id": "c.d", "kind": "widget", "label": "D"},
            {
                "id": "e",
                "kind": "decision",
                "label": "E?",
                "next": [
                    {"to": "a", "when": "yes"},
                    {"to": "b", "when": "yes"},
                ],
            },
        ]
    }
    problems = workflow_problems(broken)
    assert "Duplicate node ID 'a'" in problems
    assert "b: next node 'missing' not found" in problems
    assert "c.d: unknown kind 'widget'" in problems
    assert "c.d: parent 'c' is not a group" in problems
    assert any(p.startswith("e: repeated outcome") for p in problems)


def test_code_references_resolve():
    """Each node's `code` names a real function, class, or method"""
    for node in load_workflow()["nodes"]:
        if "code" not in node:
            continue
        path, _, qualname = node["code"].partition(":")
        assert (REPO_DIR / path).exists(), node["id"]
        assert qualname in _defined_names(REPO_DIR / path), node["id"]


def test_steps_tagged_in_code_are_in_the_map():
    """event_step() IDs in the code all name workflow nodes"""
    nodes = workflow_nodes()
    found = 0
    for path, call in _source_calls("event_step"):
        step = _step_id(path, call.args[0]) if call.args else None
        if step is None:
            continue
        found += 1
        assert step in nodes, f"{path}:{call.lineno} {step!r}"
    assert found > 20


def test_branches_recorded_in_code_are_in_the_map():
    """emit_branch() names a decision and one of its outcomes"""
    nodes = workflow_nodes()
    found = 0
    calls = [*_source_calls("emit_branch"), *_source_calls("_branch")]
    for path, call in calls:
        if path.name == "event_log.py" or not call.args:
            continue  # the helpers themselves
        step = _step_id(path, call.args[0])
        assert step is not None, f"{path}:{call.lineno}: use a literal ID"
        assert step in nodes, f"{path}:{call.lineno} {step!r}"
        found += 1
        outcomes = node_outcomes(nodes[step])
        for value in _possible_outcomes(call.args[1]):
            assert value in outcomes, f"{path}:{call.lineno} {value!r}"
    assert found > 30


def _possible_outcomes(arg):
    """Literal outcomes of an emit_branch argument (incl. a if/else)"""
    if isinstance(arg, ast.Constant):
        return [arg.value]
    if isinstance(arg, ast.IfExp):
        return _possible_outcomes(arg.body) + _possible_outcomes(arg.orelse)
    pytest.fail(f"Outcome at line {arg.lineno} is not a literal")
    return []


def test_every_collection_step_has_a_node():
    """Collection steps are tagged as collect.<step name>"""
    nodes = workflow_nodes()
    for step in COMPASSDocumentCollectionStep:
        assert f"collect.{step}" in nodes


def test_listed_events_are_emitted_somewhere():
    """Event types a node lists are recorded somewhere in the package"""
    emitted = {"branch", "workflow_graph"}
    for name in ("emit_event", "emit", "_event"):
        for _, call in _source_calls(name):
            args = _literal_args(call, 1)
            if args is not None:
                emitted.add(args[0])
    for node in load_workflow()["nodes"]:
        for kind in node.get("events", []):
            assert kind in emitted, f"{node['id']}: {kind!r}"
