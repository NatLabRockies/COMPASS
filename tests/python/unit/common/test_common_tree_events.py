"""Tests for decision-tree events in jurisdiction event logs"""

import json
from pathlib import Path

import pytest

from compass.common.base import (
    llm_response_starts_with_yes,
    setup_graph_no_nodes,
)
from compass.common.tree import AsyncDecisionTree, describe_graph
from compass.llm.calling import ChatLLMCaller
from compass.utilities.event_log import EventLog, use_event_log


class FakeService:
    """LLM service that replies with a fixed sequence of answers"""

    model_name = "test-model"

    def __init__(self, answers):
        self.answers = list(answers)

    async def call(self, **_kwargs):
        return self.answers.pop(0)


def build_tree(answers):
    """Two-question tree: continue on 'Yes', then a final answer"""
    graph = setup_graph_no_nodes(
        d_tree_name="URL check",
        chat_llm_caller=ChatLLMCaller(
            FakeService(answers), system_message="system"
        ),
    )
    graph.add_node("init", prompt="Does the URL mention the state?")
    graph.add_edge("init", "final", condition=llm_response_starts_with_yes)
    graph.add_node("final", prompt="Summarize as JSON.")
    return AsyncDecisionTree(graph)


def read_events(path):
    """Load every event from a JSONL log"""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]


@pytest.mark.asyncio
async def test_completed_tree_records_steps_and_outcome(tmp_path):
    """Each turn says whether it continued; the run says how it ended"""
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        out = await build_tree(["Yes, it does.", '{"ok": true}']).async_run()
    assert out == '{"ok": true}'
    events = read_events(log.path)
    steps = [e for e in events if e["type"] == "decision_tree_step"]
    assert [(e["node"], e["outcome"], e["next_node"]) for e in steps] == [
        ("init", "continue", "final"),
        ("final", "final", None),
    ]
    (finished,) = [e for e in events if e["type"] == "decision_tree_finished"]
    assert finished["outcome"] == "completed"
    assert finished["output"] == '{"ok": true}'
    assert finished["tags"]["thread"] == "URL check"
    requests = [e for e in events if e["type"] == "llm_request"]
    assert {e["conversation"] for e in requests} == {finished["conversation"]}


@pytest.mark.asyncio
async def test_unmatched_answer_stops_the_tree(tmp_path):
    """An answer no edge accepts ends the thread as 'stopped'"""
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        out = await build_tree(["No, it does not."]).async_run()
    assert out is None
    events = read_events(log.path)
    (step,) = [e for e in events if e["type"] == "decision_tree_step"]
    assert step["outcome"] == "stopped"
    (finished,) = [e for e in events if e["type"] == "decision_tree_finished"]
    assert finished["outcome"] == "stopped"
    assert finished["steps"] == 1


@pytest.mark.asyncio
async def test_tree_records_its_whole_graph_up_front(tmp_path):
    """The start event links the conversation to every possible path"""
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        await build_tree(["No, it does not."]).async_run()
    events = read_events(log.path)
    assert [e["type"] for e in events[:3]] == [
        "decision_tree_graph",
        "decision_tree_started",
        "llm_request",
    ]
    graph, started, request = events[:3]
    assert started["graph_id"] == graph["graph_id"]
    assert started["start"] == "init"
    assert started["conversation"] == request["conversation"]
    assert started["tags"]["thread"] == "URL check"
    # The run stopped at "init", but the graph still shows "final"
    assert graph["nodes"] == [
        {"id": "init", "prompt": "Does the URL mention the state?"},
        {"id": "final", "prompt": "Summarize as JSON."},
    ]
    assert graph["edges"] == [
        {
            "source": "init",
            "target": "final",
            "condition": "llm_response_starts_with_yes",
        }
    ]


@pytest.mark.asyncio
async def test_trees_asking_the_same_questions_share_one_graph(tmp_path):
    """A graph is written once, however many trees use it"""
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        await build_tree(["Yes.", "{}"]).async_run()
        await build_tree(["No."]).async_run()
    events = read_events(log.path)
    (graph,) = [e for e in events if e["type"] == "decision_tree_graph"]
    starts = [e for e in events if e["type"] == "decision_tree_started"]
    assert [e["graph_id"] for e in starts] == [graph["graph_id"]] * 2
    assert starts[0]["conversation"] != starts[1]["conversation"]


def test_unconditional_edges_are_described_without_a_condition():
    """An edge with no condition is the branch taken otherwise"""
    graph = setup_graph_no_nodes(d_tree_name="Else check")
    graph.add_node("init", prompt="Q1 {YES_NO_PROMPT}")
    graph.add_edge("init", "next", condition=llm_response_starts_with_yes)
    graph.add_edge("init", "final")
    graph.add_node("next", prompt="Q2")
    graph.add_node("final", prompt="Done")
    description = describe_graph(graph)
    assert [n["id"] for n in description["nodes"]] == [
        "init",
        "next",
        "final",
    ]
    assert description["nodes"][0]["prompt"] == "Q1 {YES_NO_PROMPT}"
    assert [e["condition"] for e in description["edges"]] == [
        "llm_response_starts_with_yes",
        None,
    ]
