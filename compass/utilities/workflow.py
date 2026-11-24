"""Map of a jurisdiction's run, linking event logs to its steps

The map (``compass/data/workflow.json5``) lists every step, decision,
LLM call, and decision tree of a jurisdiction's run as a graph. Events
name the node they happened in with a ``step`` tag, and decisions record
which way they went as ``branch`` events, so a replay of the log can
show the path taken against every path that was possible.
"""

import importlib.resources
from functools import cache

import pyjson5


WORKFLOW_FP = importlib.resources.files("compass") / "data" / "workflow.json5"
"""Path to the workflow map"""

NODE_KINDS = frozenset({"group", "step", "decision", "llm", "tree", "end"})
"""Kinds of node the workflow map may contain"""


@cache
def load_workflow():
    """dict: The workflow map, as parsed from its JSON5 file"""
    return pyjson5.decode(WORKFLOW_FP.read_text(encoding="utf-8"))


def workflow_nodes(workflow=None):
    """dict: Workflow nodes by ID, in the order they are listed"""
    workflow = workflow or load_workflow()
    return {node["id"]: node for node in workflow["nodes"]}


def node_edges(node):
    """list: ``(target, when)`` pairs for each edge leaving `node`

    ``when`` is ``None`` for a node with a single, unlabeled successor.
    """
    following = node.get("next")
    if following is None:
        return []
    if isinstance(following, str):
        return [(following, None)]
    return [(edge["to"], edge["when"]) for edge in following]


def node_outcomes(node):
    """list of str: The outcomes a decision at `node` can record"""
    return [when for _, when in node_edges(node) if when is not None]


def parent_id(node_id):
    """str or None: ID of the group holding the node `node_id`"""
    return node_id.rpartition(".")[0] or None


def workflow_problems(workflow=None):
    """Check that the workflow map is consistent

    Parameters
    ----------
    workflow : dict, optional
        Workflow map to check. By default, ``None``, which checks the
        packaged map.

    Returns
    -------
    list of str
        One message per problem; empty if the map is consistent.
    """
    workflow = workflow or load_workflow()
    problems = []
    nodes = {}
    for node in workflow["nodes"]:
        if node["id"] in nodes:
            problems.append(f"Duplicate node ID {node['id']!r}")
        nodes[node["id"]] = node

    for node_id, node in nodes.items():
        problems.extend(_node_problems(node_id, node, nodes))
    return problems


def _node_problems(node_id, node, nodes):
    """Problems with one node of the workflow map"""
    return _field_problems(node_id, node) + _link_problems(
        node_id, node, nodes
    )


def _field_problems(node_id, node):
    """Problems with a node's own fields"""
    problems = []
    if node.get("kind") not in NODE_KINDS:
        problems.append(f"{node_id}: unknown kind {node.get('kind')!r}")
    if not node.get("label"):
        problems.append(f"{node_id}: missing label")
    if node.get("kind") == "tree" and not node.get("tree"):
        problems.append(f"{node_id}: tree node without a tree name")
    if node.get("tone") not in {None, "good", "bad"}:
        problems.append(f"{node_id}: unknown tone {node['tone']!r}")
    return problems


def _link_problems(node_id, node, nodes):
    """Problems with a node's parent, shared group, and edges"""
    problems = []
    parent = parent_id(node_id)
    if parent is not None and nodes.get(parent, {}).get("kind") != "group":
        problems.append(f"{node_id}: parent {parent!r} is not a group")

    uses = node.get("uses")
    if uses is not None and nodes.get(uses, {}).get("kind") != "group":
        problems.append(f"{node_id}: uses {uses!r}, which is not a group")

    targets = node_edges(node)
    for target, _ in targets:
        if target not in nodes:
            problems.append(f"{node_id}: next node {target!r} not found")
    outcomes = node_outcomes(node)
    if len(outcomes) != len(set(outcomes)):
        problems.append(f"{node_id}: repeated outcome in {outcomes}")
    if isinstance(node.get("next"), list) and len(outcomes) != len(targets):
        problems.append(f"{node_id}: every listed edge needs a 'when'")
    return problems
