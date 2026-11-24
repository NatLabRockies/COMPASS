"""Ordinance async decision tree"""

import logging
from functools import cached_property

import networkx as nx
from elm.tree import DecisionTree

from compass.utilities.enums import LLMUsageCategory
from compass.exceptions import COMPASSRuntimeError
from compass.utilities.event_log import (
    current_event_log,
    emit_event,
    error_details,
    event_tags,
)


logger = logging.getLogger(__name__)


class AsyncDecisionTree(DecisionTree):
    """Async class to traverse a directed graph of LLM prompts

    Nodes of this tree are prompts, and edges are transitions between
    prompts based on conditions being met in the LLM response

    Purpose:
        Represent a series of prompts that can be used in sequence to
        extract values of interest from text.
    Responsibilities:
        1. Store all prompts used to extract a particular ordinance
           value from text.
        2. Track relationships between the prompts (i.e. which prompts
           is used first, which prompt is used next depending on the
           output of the previous prompt, etc.) using a directed acyclic
           graph.
    Key Relationships:
        Inherits from :class:`~elm.tree.DecisionTree` to add ``async``
        capabilities. Uses a ChatLLMCaller for LLm queries.
    """

    def __init__(self, graph, usage_sub_label=None):
        """

        Parameters
        ----------
        graph : networkx.DiGraph
            Directed acyclic graph where nodes are LLM prompts and edges
            are logical transitions based on the response. Must have
            high-level graph attribute "chat_llm_caller" which is a
            ChatLLMCaller instance. Nodes should have attribute "prompt"
            which can have {format} named arguments that will be filled
            from the high-level graph attributes. Edges can have
            attribute "condition" that is a callable to be executed on
            the LLM response text. An edge from a node without a
            condition acts as an "else" statement if no other edge
            conditions are satisfied. A single edge from node to node
            does not need a condition.
        usage_sub_label : str, optional
            Optional label to classify LLM usage under when running this
            decision tree. If ``None``, will simply label calls made
            from this tree under "decision_tree". By default, ``None``.
        """
        self._g = graph
        self._history = []
        self.usage_sub_label = (
            usage_sub_label or LLMUsageCategory.DECISION_TREE
        )
        assert isinstance(self.graph, nx.DiGraph)
        assert "chat_llm_caller" in self.graph.graph

    @property
    def chat_llm_caller(self):
        """ChatLLMCaller: LLM caller bound to the decision tree"""
        return self.graph.graph["chat_llm_caller"]

    @cached_property
    def tree_name(self):
        """str: Configured decision tree name"""
        return self._g.graph.get("_d_tree_name", "Unknown decision tree")

    @property
    def messages(self):
        """list: Conversation messages exchanged with the LLM"""
        return self.chat_llm_caller.messages

    @property
    def all_messages_txt(self):
        """str: Formatted conversation transcript"""
        messages = [
            f"{msg['role'].upper()}: {msg['content']}" for msg in self.messages
        ]
        return "\n\n".join(messages)

    async def async_call_node(self, node0):
        """Call the LLM

        The chat will start with the prompt from the input node and will
        search the successor edges for a valid transition condition.

        Parameters
        ----------
        node0 : str
            Name of node being executed.

        Returns
        -------
        out : str
            Next node or LLM response if at a leaf node.
        """
        prompt = self._prepare_graph_call(node0)
        with event_tags(thread=self.tree_name, node=node0):
            out = await self.chat_llm_caller.call(
                prompt, usage_sub_label=self.usage_sub_label
            )
            try:
                result = self._parse_graph_output(node0, out or "")
            except AttributeError:
                self._record_step(node0, "stopped", None)
                raise
            leaf = result not in self.graph
            self._record_step(
                node0,
                "final" if leaf else "continue",
                None if leaf else result,
            )
        logger.debug_to_file(
            "Chat GPT prompt (node=%r; name=%r):\n%s\nChat GPT response:\n%s",
            node0,
            self.tree_name,
            prompt,
            out,
        )
        return result

    def _record_step(self, node, outcome, next_node):
        """Record whether a turn continued, finished, or stopped"""
        emit_event(
            "decision_tree_step",
            conversation=self._conversation_id,
            node=node,
            outcome=outcome,
            next_node=next_node,
        )

    @property
    def _conversation_id(self):
        log = current_event_log()
        if log is None:
            return None
        return self.chat_llm_caller.conversation_id(log)

    def _record_start(self, node0):
        """Record the tree's whole graph and the conversation running it

        The graph includes the questions and branches this run may never
        reach, so a replay can show the path taken against every path
        that was possible.
        """
        log = current_event_log()
        if log is None:
            return
        log.emit(
            "decision_tree_started",
            conversation=self.chat_llm_caller.conversation_id(log),
            graph_id=log.graph_id(describe_graph(self.graph)),
            start=node0,
        )

    async def async_run(self, node0="init"):
        """Traverse the decision tree, recording how it ends"""
        with event_tags(thread=self.tree_name):
            self._record_start(node0)
            try:
                out = await self._async_run(node0)
            except Exception as exc:
                emit_event(
                    "decision_tree_finished",
                    conversation=self._conversation_id,
                    outcome="error",
                    **error_details(exc),
                    steps=len(self._history),
                )
                raise
            emit_event(
                "decision_tree_finished",
                conversation=self._conversation_id,
                outcome="stopped" if out is None else "completed",
                output=out,
                steps=len(self._history),
            )
        return out

    async def _async_run(self, node0="init"):
        """Traverse the decision tree starting at the input node

        Parameters
        ----------
        node0 : str
            Name of starting node in the graph. This is typically called
            "init".

        Returns
        -------
        out : str or None
            Final response from LLM at the leaf node or ``None`` if an
            ``AttributeError`` was raised during execution.

        Raises
        ------
        compass.exceptions.COMPASSRuntimeError
            Raised when the traversal encounters an unexpected
            exception that is not an ``AttributeError``.
        """

        self._history = []

        while True:
            try:
                out = await self.async_call_node(node0)
            except AttributeError:
                logger.debug_to_file(
                    "Error traversing trees, here's the full "
                    "conversation printout:\n%s",
                    self.all_messages_txt,
                )
                return None
            except Exception as e:
                logger.debug_to_file(
                    "Error traversing trees, here's the full "
                    "conversation printout:\n%s",
                    self.all_messages_txt,
                )
                last_message = self.messages[-1]["content"]
                msg = (
                    "Ran into an exception when traversing tree. "
                    "Last message from LLM is printed below. "
                    "See debug logs for more detail. "
                    "\nLast message: \n"
                    '"""\n%s\n"""'
                )
                logger.exception(msg, last_message)
                raise COMPASSRuntimeError(msg % last_message) from e

            if out in self.graph:
                node0 = out
            else:
                break

        logger.info(
            "Final decision tree output (name=%r): %s", self.tree_name, out
        )

        return out


def describe_graph(graph):
    """Questions and branches of a decision-tree graph, for event logs

    Prompts are kept as templates (placeholders such as ``{text}`` are
    left unfilled), so the description stays small and is the same for
    every document the tree runs on.

    Parameters
    ----------
    graph : networkx.DiGraph
        Decision-tree graph whose nodes have a "prompt" attribute and
        whose edges may have a "condition" callable.

    Returns
    -------
    dict
        ``nodes`` (each with an ``id`` and ``prompt``) and ``edges``
        (each with a ``source``, ``target``, and ``condition``), in the
        order the tree checks them. ``condition`` is the condition
        function's name, or ``None`` for an edge taken when no other
        condition matches.
    """
    return {
        "nodes": [
            {"id": node, "prompt": attrs.get("prompt")}
            for node, attrs in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": source,
                "target": target,
                "condition": _condition_name(attrs.get("condition")),
            }
            for source, target, attrs in graph.edges(data=True)
        ],
    }


def _condition_name(condition):
    """Name of an edge condition, or ``None`` if the edge has none"""
    if condition is None:
        return None
    return getattr(condition, "__name__", repr(condition))
