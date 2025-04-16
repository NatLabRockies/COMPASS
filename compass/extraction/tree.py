"""Ordinance async decision tree"""

import networkx as nx
import logging

from elm.tree import DecisionTree

from compass.utilities.enums import LLMUsageCategory
from compass.exceptions import COMPASSRuntimeError


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
        capabilities. Uses a
        :class:`~compass.llm.calling.ChatLLMCaller` for LLm queries.
    """

    def __init__(self, graph):
        """

        Parameters
        ----------
        graph : nx.DiGraph
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
        """
        self._g = graph
        self._history = []
        assert isinstance(self.graph, nx.DiGraph)
        assert "chat_llm_caller" in self.graph.graph

    @property
    def chat_llm_caller(self):
        """ChatLLMCaller: ChatLLMCaller instance for this tree"""
        return self.graph.graph["chat_llm_caller"]

    @property
    def messages(self):
        """Get a list of the conversation messages with the LLM

        Returns
        -------
        list
        """
        return self.chat_llm_caller.messages

    @property
    def all_messages_txt(self):
        """Get a printout of the full conversation with the LLM

        Returns
        -------
        str
        """
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
        out = await self.chat_llm_caller.call(
            prompt, usage_sub_label=LLMUsageCategory.DECISION_TREE
        )
        logger.debug_to_file(
            "Chat GPT prompt:\n%s\nChat GPT response:\n%s", prompt, out
        )
        return self._parse_graph_output(node0, out or "")

    async def async_run(self, node0="init"):
        """Traverse the decision tree starting at the input node

        Parameters
        ----------
        node0 : str
            Name of starting node in the graph. This is typically called
            "init".

        Returns
        -------
        out : str | None
            Final response from LLM at the leaf node or ``None`` if an
            ``AttributeError`` was raised during execution.
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

        logger.info("Final decision tree output: %s", out)

        return out
