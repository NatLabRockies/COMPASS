"""Common ordinance extraction components"""

import logging

import networkx as nx

from compass.utilities import llm_response_as_json
from compass.extraction.tree import AsyncDecisionTree


logger = logging.getLogger(__name__)
_SECTION_PROMPT = (
    'The value of the "section" key should be a string representing the '
    "title of the section (including numerical labels), if it's given, "
    "and `null` otherwise."
)
_COMMENT_PROMPT = (
    'The value of the "comment" key should be a one-sentence explanation '
    "of how you determined the value, if you think it is necessary "
    "(`null` otherwise)."
)
_SUMMARY_PROMPT = (
    'The value of the "summary" key should be a two-sentence short summary '
    "of the ordinance, quoting the text directly if possible."
)
EXTRACT_ORIGINAL_TEXT_PROMPT = (
    "Can you extract the raw text with original formatting "
    "that states how close I can site {tech} to {feature}? "
)


def setup_graph_no_nodes(**kwargs):
    """Setup a graph with no nodes

    This function is used to set keywords on the graph that can be used
    in text prompts on the graph nodes.

    Returns
    -------
    nx.DiGraph
        Graph with no nodes but with global keywords set.
    """
    return nx.DiGraph(
        SECTION_PROMPT=_SECTION_PROMPT,
        COMMENT_PROMPT=_COMMENT_PROMPT,
        SUMMARY_PROMPT=_SUMMARY_PROMPT,
        **kwargs,
    )


def llm_response_starts_with_yes(response):
    """Check if LLM response begins with "yes" (case-insensitive)

    Parameters
    ----------
    response : str
        LLM response string.

    Returns
    -------
    bool
        `True` if LLM response begins with "Yes".
    """
    return response.lower().startswith("yes")


def llm_response_starts_with_no(response):
    """Check if LLM response begins with "no" (case-insensitive)

    Parameters
    ----------
    response : str
        LLM response string.

    Returns
    -------
    bool
        `True` if LLM response begins with "No".
    """
    return response.lower().startswith("no")


def llm_response_does_not_start_with_no(response):
    """Check if LLM response does not start with "no" (case-insensitive)

    Parameters
    ----------
    response : str
        LLM response string.

    Returns
    -------
    bool
        `True` if LLM response does not begin with "No".
    """
    return not llm_response_starts_with_no(response)


def setup_async_decision_tree(graph_setup_func, **kwargs):
    """Setup Async Decision tree for ordinance extraction"""
    G = graph_setup_func(**kwargs)  # noqa: N806
    tree = AsyncDecisionTree(G)
    assert len(tree.chat_llm_caller.messages) == 1
    return tree


def found_ord(messages):
    """Check messages from the base graph to see if ordinance was found

    ..IMPORTANT:: This function may break if the base graph structure
                  changes. Always update the hardcoded values to match
                  the base graph message containing the LLM response
                  about ordinance content.
    """
    num_messages_in_base_tree = 3
    if len(messages) < num_messages_in_base_tree:
        return False
    return llm_response_starts_with_yes(messages[2].get("content", ""))


async def run_async_tree(tree, response_as_json=True):
    """Run Async Decision Tree and return output as dict"""
    try:
        response = await tree.async_run()
    except RuntimeError:
        msg = (
            "    - NOTE: This is not necessarily an error and may just mean "
            "that the text does not have the requested data."
        )
        logger.error(msg)  # noqa: TRY400
        response = None

    if response_as_json:
        return llm_response_as_json(response) if response else {}

    return response


async def run_async_tree_with_bm(tree, base_messages):
    """Run Async Decision Tree from base messages; return dict output"""
    tree.chat_llm_caller.messages = base_messages
    assert len(tree.chat_llm_caller.messages) == len(base_messages)
    return await run_async_tree(tree)


def empty_output(feature):
    """Empty output for a feature (not found in text)"""
    if feature in {"struct", "prop_line"}:
        return [
            {"feature": f"{feature} (participating)"},
            {"feature": f"{feature} (non-participating)"},
        ]
    return [{"feature": feature}]


def setup_base_graph(**kwargs):
    """Setup graph to get setback ordinance text for a given feature

    Parameters
    ----------
    **kwargs
        Keyword-value pairs to add to graph.

    Returns
    -------
    nx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    G = setup_graph_no_nodes(**kwargs)  # noqa: N806

    G.add_node(
        "init",
        prompt=(
            "Is there text in the following legal document that describes "
            "how far I have to setback {tech} from {feature}? "
            "{feature_clarifications}"
            "Pay extra attention to clarifying text found in parentheses "
            "and footnotes. Begin your response with either 'Yes' or 'No' "
            "and explain your answer."
            '\n\n"""\n{text}\n"""'
        ),
    )

    G.add_edge(
        "init", "get_text", condition=llm_response_does_not_start_with_no
    )
    G.add_node("get_text", prompt=EXTRACT_ORIGINAL_TEXT_PROMPT)

    return G


def setup_participating_owner(**kwargs):
    """Setup graph to check for "participating" setbacks for a feature

    Parameters
    ----------
    **kwargs
        Keyword-value pairs to add to graph.

    Returns
    -------
    nx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    G = setup_graph_no_nodes(**kwargs)  # noqa: N806

    G.add_node(
        "init",
        prompt=(
            "Does the ordinance for {feature} setbacks explicitly specify "
            "a value that applies to participating owners? Occupying owners "
            "are not participating owners unless explicitly mentioned in the "
            "text. Justify your answer by quoting the raw text directly."
        ),
    )
    G.add_edge("init", "non_part")
    G.add_node(
        "non_part",
        prompt=(
            "Does the ordinance for {feature} setbacks explicitly specify "
            "a value that applies to non-participating owners? Non-occupying "
            "owners are not non-participating owners unless explicitly "
            "mentioned in the text. Justify your answer by quoting the raw "
            "text directly."
        ),
    )
    G.add_edge("non_part", "final")
    G.add_node(
        "final",
        prompt=(
            "Now we are ready to extract structured data. Respond based on "
            "our entire conversation so far. Return your answer in JSON "
            "format (not markdown). Your JSON file must include exactly two "
            'keys. The keys are "participating" and "non-participating". The '
            'value of the "participating" key should be a string containing '
            "the raw text with original formatting from the ordinance that "
            "applies to participating owners or `null` if there was no such "
            'text. The value of the "non-participating" key should be a '
            "string containing the raw text with original formatting from the "
            "ordinance that applies to non-participating owners or simply the "
            "full ordinance if the text did not make the distinction between "
            "participating and non-participating owners."
        ),
    )
    return G


def setup_graph_extra_restriction(is_numerical=True, **kwargs):
    """Setup Graph to extract non-setback ordinance values from text

    Parameters
    ----------
    **kwargs
        Keyword-value pairs to add to graph.

    Returns
    -------
    nx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    G = setup_graph_no_nodes(**kwargs)  # noqa: N806

    G.add_node(
        "init",
        prompt=(
            "We will attempt to extract structured data for this ordinance. "
            "Does the following text explicitly mention {restriction} for "
            "{tech}? Do not infer based on other restrictions; if this "
            "particular restriction is not explicitly mentioned then say "
            "'No'. Pay extra attention to clarifying text found in "
            "parentheses and footnotes. Begin your response with either "
            "'Yes' or 'No' and explain your answer."
            '\n\n"""\n{text}\n"""'
        ),
    )
    G.add_edge("init", "final", condition=llm_response_starts_with_yes)

    if is_numerical:
        G.add_node(
            "final",
            prompt=(
                "Now we are ready to extract structured data. Respond based "
                "on our entire conversation so far. Return your answer in "
                "JSON format (not markdown). Your JSON file must include "
                'exactly five keys. The keys are "value", "units", "summary", '
                '"section", and "comment". The value of the "value" key '
                "should be a numerical value corresponding to the "
                "{restriction} for {tech}, or `null` if the text does not "
                "mention such a restriction. Use our conversation to fill "
                'out this value. The value of the "units" key should be a '
                "string corresponding to the units for the {restriction} "
                "allowed for {tech} by the text below, or `null` if the text "
                "does not mention such a restriction. Make sure to include "
                'any "per XXX" clauses in the units. {SUMMARY_PROMPT} '
                "{SECTION_PROMPT} {COMMENT_PROMPT}"
            ),
        )
    else:
        G.add_node(
            "final",
            prompt=(
                "Now we are ready to extract structured data. Respond based "
                "on our entire conversation so far. Return your answer in "
                "JSON format (not markdown). Your JSON file must include "
                'exactly three keys. The keys are "summary", "section", and '
                '"comment". {SUMMARY_PROMPT} {SECTION_PROMPT} {COMMENT_PROMPT}'
            ),
        )
    return G
