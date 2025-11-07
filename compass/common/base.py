"""Common ordinance extraction components"""

import asyncio
import logging
from datetime import datetime

import networkx as nx
from elm import ApiBase

from compass.common.tree import AsyncDecisionTree
from compass.utilities import llm_response_as_json
from compass.utilities.enums import LLMUsageCategory
from compass.utilities.parsing import (
    merge_overlapping_texts,
    clean_backticks_from_llm_response,
)
from compass.exceptions import COMPASSRuntimeError


logger = logging.getLogger(__name__)
_SECTION_PROMPT = (
    "The value of the 'section' key should be a string representing the "
    "title of the section (including numerical labels), if it's given, "
    "and `null` otherwise."
)
_SUMMARY_PROMPT = (
    "The value of the 'summary' key should be a short summary of the relevant "
    "ordinance, **using direct text excerpts as much as possible.** "
    "If you had to make a selection when reporting the ordinance, be sure to "
    "list out all the other options and their conditions in the summary."
)
_UNITS_IN_SUMMARY_PROMPT = (
    "Also include any clarifications about the units in the summary."
)
EXTRACT_ORIGINAL_SETBACK_TEXT_PROMPT = (
    "Extract all portions of the text (with original formatting) "
    "that state how close I can site {tech} to {feature}. "
    "{feature_clarifications}"
    "Please consider only ordinances relating to setbacks from {feature}; "
    "do not respond based on any text related to {ignore_features}. "
    "The extracted text will be used for structured data extraction, so it "
    "must be both **comprehensive** (retaining all relevant details) and "
    "**focused** (excluding unrelated content). Ensure that all retained "
    "information is **directly applicable** to {system_size_reminder}"
)


def setup_graph_no_nodes(d_tree_name="Unknown Decision Tree", **kwargs):
    """Setup a graph with no nodes

    This function is used to set keywords on the graph that can be used
    in text prompts on the graph nodes.

    Parameters
    ----------
    d_tree_name : str, default="Unknown Decision Tree"
        Name of the decision tree being set up (used for logging).
        By default, "Unknown Decision Tree".
    **kwargs
        Keyword-value pairs to add to graph.

    Returns
    -------
    networkx.DiGraph
        Graph with no nodes but with global keywords set.
    """
    feat = kwargs.get("feature_id", kwargs.get("feature", kwargs.get("url")))
    if feat:
        d_tree_name = f"{d_tree_name} ({feat})"

    return nx.DiGraph(
        SECTION_PROMPT=_SECTION_PROMPT,
        SUMMARY_PROMPT=_SUMMARY_PROMPT,
        UNITS_IN_SUMMARY_PROMPT=_UNITS_IN_SUMMARY_PROMPT,
        _d_tree_name=d_tree_name,
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


def setup_async_decision_tree(
    graph_setup_func, usage_sub_label=None, **kwargs
):
    """Setup Async Decision tree for ordinance extraction"""
    G = graph_setup_func(**kwargs)  # noqa: N806
    tree = AsyncDecisionTree(G, usage_sub_label=usage_sub_label)
    assert len(tree.chat_llm_caller.messages) == 1
    return tree


async def run_async_tree(tree, response_as_json=True):
    """Run Async Decision Tree and return output as dict"""
    try:
        response = await tree.async_run()
    except COMPASSRuntimeError:
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
    if feature in {"structures", "property line"}:
        return [
            {"feature": f"{feature} (participating)"},
            {"feature": f"{feature} (non-participating)"},
        ]
    return [{"feature": feature}]


def setup_base_setback_graph(**kwargs):
    """Setup graph to get setback ordinance text for a given feature

    Parameters
    ----------
    **kwargs
        Keyword-value pairs to add to graph.

    Returns
    -------
    networkx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    G = setup_graph_no_nodes(  # noqa: N806
        d_tree_name="Base setback questions", **kwargs
    )

    G.add_node(
        "init",
        prompt=(
            "Is there text in the following legal document that describes "
            "how far I have to setback {tech} from {feature}? "
            "{feature_clarifications}"  # expected to end in space
            "Please consider only setbacks from {feature}. "
            "Please also only consider setbacks that would apply for "
            "{system_size_reminder}"  # expected to end in space
            "Don't forget to pay extra attention to clarifying text found "
            "in parentheses and footnotes. "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain your answer."
            '\n\n"""\n{text}\n"""'
        ),
    )

    G.add_edge(
        "init", "verify_feature", condition=llm_response_does_not_start_with_no
    )
    G.add_node(
        "verify_feature",
        prompt=(
            "Did you infer your answer based on setback requirements from "
            "something other than {feature}, such as {ignore_features}? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain your answer."
        ),
    )
    if "roads" in kwargs.get("feature", ""):
        G.add_edge(
            "verify_feature",
            "check_if_property_line",
            condition=llm_response_starts_with_no,
        )
        G.add_node(
            "check_if_property_line",
            prompt=(
                "Is this requirement better classified as a setback from "
                "property lines? Please start your response with "
                "either 'Yes' or 'No' and briefly explain your answer."
            ),
        )
        G.add_edge(
            "check_if_property_line",
            "get_text",
            condition=llm_response_starts_with_no,
        )
    else:
        G.add_edge(
            "verify_feature", "get_text", condition=llm_response_starts_with_no
        )

    G.add_node("get_text", prompt=EXTRACT_ORIGINAL_SETBACK_TEXT_PROMPT)

    return G


def setup_participating_owner(**kwargs):
    """Setup graph to check for "participating" setbacks for a feature

    Parameters
    ----------
    **kwargs
        Keyword-value pairs to add to graph.

    Returns
    -------
    networkx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    G = setup_graph_no_nodes(  # noqa: N806
        d_tree_name="Participating owner", **kwargs
    )

    G.add_node(
        "init",
        prompt=(
            "Does the ordinance for {feature} setbacks explicitly distinguish "
            "between **participating** and **non-participating** {owned_type} "
            "owners? Participating may, for example, be able to sign a waiver "
            "or enter some sort of agreement to reduce or completely "
            "eliminate the setback requirement. "
            "{feature_clarifications} We are only interested in "
            "setbacks from {feature}; do not base your response on any text "
            "related to {ignore_features}. "
            "Please only consider setbacks that would apply for "
            "{system_size_reminder}"
            "Please start your response with either 'Yes' or 'No' "
            "and briefly explain your answer."
        ),
    )
    G.add_edge("init", "waiver", condition=llm_response_starts_with_yes)
    G.add_node(
        "waiver",
        prompt=(
            "Does the ordinance allow **participating** {owned_type} owners "
            "to completely waive or reduce by an unspecified amount the "
            "{feature} setbacks requirements? "
            "Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
        ),
    )

    G.add_edge("waiver", "p_same_as_np", condition=llm_response_starts_with_no)
    G.add_node(
        "p_same_as_np",
        prompt=(
            "Does the ordinance for {feature} setbacks explicitly specify "
            "that **participating** {owned_type} owners must abide to the "
            "same setbacks as **non-participating** {owned_type} owners? "
            "Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
        ),
    )

    G.add_edge(
        "p_same_as_np",
        "final_p_same_as_np",
        condition=llm_response_starts_with_yes,
    )
    G.add_node(
        "final_p_same_as_np",
        prompt=(
            "Please respond based on our entire conversation so far. "
            "Return your answer as a single dictionary in JSON format (not "
            "markdown). Your JSON file must include exactly one key. The "
            "key is 'participating'. The value of the 'participating' key "
            "should be a string containing the raw text with original "
            "formatting from the ordinance that applies to both "
            "**participating** and **non-participating** owners. Be sure to "
            "include the numerical value for {feature} setbacks that these "
            "owners must abide by."
        ),
    )

    G.add_edge("p_same_as_np", "part", condition=llm_response_starts_with_no)
    G.add_node(
        "part",
        prompt=(
            "Does the ordinance for {feature} setbacks explicitly specify "
            "a **numerical** value that applies to **participating** "
            "{owned_type} owners? "
            "Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
        ),
    )
    G.add_edge("part", "non_part", condition=llm_response_starts_with_yes)
    G.add_node(
        "non_part",
        prompt=(
            "Does the ordinance for {feature} setbacks explicitly specify "
            "a **numerical** value that applies to **non-participating** "
            "{owned_type} owners? "
            "If your answer is 'yes', justify it by quoting the raw text "
            "directly."
        ),
    )
    G.add_edge("non_part", "final")
    G.add_node(
        "final",
        prompt=(
            "Please respond based on our entire conversation so far. "
            "Return your answer as a single dictionary in JSON format (not "
            "markdown). Your JSON file must include exactly two keys. The "
            "keys are 'participating' and 'non-participating'. The value of "
            "the 'participating' key should be a string containing the raw "
            "text with original formatting from the ordinance that applies to "
            "**participating** owners. The value of the 'non-participating' "
            "key should be a string containing the raw text with original "
            "formatting from the ordinance that applies to "
            "**non-participating** owners for {feature} setbacks."
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
    networkx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    kwargs.setdefault("unit_clarification", "")
    kwargs.setdefault("feature_clarifications", "")
    feature_id = kwargs.get("feature_id", "")
    G = setup_graph_no_nodes(  # noqa: N806
        d_tree_name="Extra restriction", **kwargs
    )

    G.add_node(
        "init",
        prompt=(
            "Does the following legal text explicitly enact {restriction} for "
            "{tech} that an energy system developer would have to abide to? "
            "{feature_clarifications}\n"
            "Make sure your answer adheres to these guidelines:\n"
            "1) Respond based only on the explicit text provided for "
            "{restriction}. Do not infer or assume relevance based on general "
            "definitions, interpretations, or overlap with other categories. "
            "Do not include content just because it could be considered "
            "related to {restriction} by definition. If the text does not "
            "directly mention or clearly describe {restriction} for {tech}, "
            "respond with 'No'.\n"
            "2) If the text only provides a definition of what {restriction} "
            "are without providing specifics, please respond with 'No'.\n"
            "3) Please focus only on {restriction} that would apply for "
            "{system_size_reminder}\n"
            "4) Pay close attention to clarifying details in parentheses, "
            "footnotes, or additional explanatory text.\n"
            "5) Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
            '\n\n"""\n{text}\n"""'
        ),
    )

    if is_numerical:
        if "other" in feature_id:
            _add_other_system_setback_clarification_nodes(G)
        elif "coverage" in feature_id:
            _add_coverage_clarification_nodes(G)
        elif "land density" in feature_id:
            _add_land_density_clarification_nodes(G)
        elif "minimum lot size" in feature_id:
            _add_minimum_lot_size_clarification_nodes(G)
        elif "maximum lot size" in feature_id:
            _add_maximum_lot_size_clarification_nodes(G)
        elif "maximum project size" in feature_id:
            _add_maximum_project_size_clarification_nodes(G)
        elif "maximum turbine height" in feature_id:
            _add_maximum_turbine_height_clarification_nodes(G)
        else:
            G.add_edge("init", "value", condition=llm_response_starts_with_yes)

        _add_value_and_units_clarification_nodes(G)

        G.add_edge("units", "final")
        G.add_node(
            "final",
            prompt=(
                "Please respond based on our entire conversation so far. "
                "Return your answer as a dictionary in "
                "JSON format (not markdown). Your JSON file must include "
                "exactly four keys. The keys are 'value', 'units', 'summary', "
                "and 'section'. The value of the 'value' key "
                "should be a numerical value corresponding to the "
                "{restriction} for {tech}, or `null` if the text "
                "does not mention such a restriction. Use our conversation to "
                "fill out this value. The value of the 'units' key should be "
                "a string corresponding to the (standard) units for the "
                "{restriction} allowed for {tech} by the text "
                "below, or `null` if the text does not mention such a "
                "restriction. "
                "As before, focus only on {restriction} specifically for "
                "{system_size_reminder}"
                "{SUMMARY_PROMPT} {UNITS_IN_SUMMARY_PROMPT} {SECTION_PROMPT}"
            ),
        )

    elif "prohibitions" in feature_id:
        _add_prohibitions_extraction_nodes(G)

    else:
        G.add_edge("init", "final", condition=llm_response_starts_with_yes)
        G.add_node(
            "final",
            prompt=(
                "Please respond based on our entire conversation so far. "
                "Return your answer as a dictionary in "
                "JSON format (not markdown). Your JSON file must include "
                "exactly two keys. The keys are 'summary' and 'section'. "
                "{SUMMARY_PROMPT} {SECTION_PROMPT}"
            ),
        )

    G.add_edge("init", "enr", condition=llm_response_starts_with_no)
    G.add_node(
        "enr",
        prompt=(
            "Does the legal text **directly** mention not regulating "
            "{restriction} for {tech}? "
            "As before, focus only on {restriction} specifically for "
            "{system_size_reminder}"
            "Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
        ),
    )
    G.add_edge("enr", "enr_text", condition=llm_response_starts_with_yes)
    G.add_node(
        "enr_text",
        prompt=(
            "What is the **exact text excerpt** that explicitly states there "
            "are no regulations around {restriction} for {tech}? "
        ),
    )
    G.add_edge("enr_text", "enr_validate")
    G.add_node(
        "enr_validate",
        prompt=(
            "Based on your response, are you still confident that the text "
            "**directly** states that there are no regulations around "
            "{restriction} for {tech}? "
            "Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
        ),
    )
    G.add_edge(
        "enr_validate", "final_enr", condition=llm_response_starts_with_yes
    )
    G.add_node(
        "final_enr",
        prompt=(
            "Please respond based on our entire conversation so far. "
            "Return your answer as a dictionary in JSON format (not "
            "markdown). Your JSON file must include exactly three keys. "
            "The keys are 'value', 'summary', and 'section'. The value of "
            "the 'value' key should be the string 'ENR' if the text "
            "clearly states that the jurisdiction does not regulate "
            "{restriction} for {tech} or `null` otherwise. "
            "{SUMMARY_PROMPT} {SECTION_PROMPT}"
        ),
    )

    return G


def _add_other_system_setback_clarification_nodes(G):  # noqa: N803
    """Add nodes and edges to clarify "other system" setbacks"""
    G.add_edge("init", "is_intra_farm", condition=llm_response_starts_with_yes)
    G.add_node(
        "is_intra_farm",
        prompt=(
            "Does the separation requirement apply to full farms "
            "and/or utility-size installations? If so, please start "
            "your answer with 'Yes'. If the separation requirement "
            "only applies to individual farm components (i.e. "
            "individual energy generation system units), please start "
            "your response with 'No'. In either case, briefly explain "
            "your answer."
        ),
    )
    G.add_edge(
        "is_intra_farm", "value", condition=llm_response_starts_with_yes
    )
    return G


def _add_coverage_clarification_nodes(G):  # noqa: N803
    """Add nodes and edges to clarify "coverage" extraction"""
    G.add_edge("init", "is_area", condition=llm_response_starts_with_yes)
    G.add_node(
        "is_area",
        prompt=(
            "Is the coverage reported as an area value? "
            "Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
        ),
    )
    G.add_edge("is_area", "value", condition=llm_response_starts_with_no)
    return G


def _add_land_density_clarification_nodes(G):  # noqa: N803
    """Add nodes and edges to clarify "land density" extraction"""
    G.add_edge(
        "init", "correct_density_units", condition=llm_response_starts_with_yes
    )
    G.add_node(
        "correct_density_units",
        prompt=(
            "Is the density reported as a system size **per area** "
            "value? "
            "Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
        ),
    )
    G.add_edge(
        "correct_density_units",
        "value",
        condition=llm_response_starts_with_yes,
    )
    return G


def _add_minimum_lot_size_clarification_nodes(G):  # noqa: N803
    """Add nodes and edges to clarify "minimum lot size" extraction"""
    G.add_edge(
        "init", "correct_min_ls_units", condition=llm_response_starts_with_yes
    )
    G.add_node(
        "correct_min_ls_units",
        prompt=(
            "Is the minimum lot size reported as an **area** value? "
            "Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
        ),
    )
    G.add_edge(
        "correct_min_ls_units", "value", condition=llm_response_starts_with_yes
    )
    return G


def _add_maximum_lot_size_clarification_nodes(G):  # noqa: N803
    """Add nodes and edges to clarify "maximum lot size" extraction"""
    G.add_edge(
        "init", "correct_max_ls_units", condition=llm_response_starts_with_yes
    )
    G.add_node(
        "correct_max_ls_units",
        prompt=(
            "Is the maximum lot size reported as an **area** value? "
            "Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
        ),
    )
    G.add_edge(
        "correct_max_ls_units", "value", condition=llm_response_starts_with_yes
    )
    return G


def _add_maximum_project_size_clarification_nodes(G):  # noqa: N803
    """Add nodes and edges to clarify "max project size" extraction"""
    G.add_edge("init", "is_mps_area", condition=llm_response_starts_with_yes)
    G.add_node(
        "is_mps_area",
        prompt=(
            "Does the project size requirement specifically provide "
            "a system size in MW or an installation size (e.g. "
            "maximum number of systems or maximum number of solar "
            "panels)? "
            "Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
        ),
    )
    G.add_edge(
        "is_mps_area",
        "is_mps_conditional",
        condition=llm_response_starts_with_yes,
    )
    G.add_node(
        "is_mps_conditional",
        prompt=(
            "Can the project size requirement be bypassed by applying "
            "for a permit? "
            "Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
        ),
    )

    G.add_edge(
        "is_mps_conditional", "value", condition=llm_response_starts_with_no
    )
    return G


def _add_maximum_turbine_height_clarification_nodes(G):  # noqa: N803
    """Add nodes and edges to clarify max turbine height extraction"""
    G.add_edge("init", "has_relative", condition=llm_response_starts_with_yes)
    G.add_node(
        "has_relative",
        prompt=(
            "Are any of the turbine height restrictions measured from a point "
            "other than the ground (e.g. a building height or an airspace "
            "level, etc.)? "
            "Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
        ),
    )
    G.add_edge(
        "has_relative",
        "has_non_relative",
        condition=llm_response_starts_with_yes,
    )
    G.add_edge("has_relative", "value", condition=llm_response_starts_with_no)

    G.add_node(
        "has_non_relative",
        prompt=(
            "We are not interested in restrictions that are relative to some "
            "level. Does the legal text enact any turbine height restrictions "
            "measured from the ground? "
            "Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
        ),
    )

    G.add_edge(
        "has_non_relative", "value", condition=llm_response_starts_with_yes
    )
    return G


def _add_value_and_units_clarification_nodes(G):  # noqa: N803
    """Add nodes and edges to clarify value and units extraction"""

    G.add_node(
        "value",
        prompt=(
            "What is the **numerical** value given for the "
            "{restriction} for {tech}? Follow these guidelines:\n"
            "1) Extract only the explicit numerical value provided for "
            "the restriction. Do not infer values from related "
            "restrictions.\n"
            "2) If multiple values are given, select the most restrictive "
            "one (i.e., the smallest allowable limit, the lowest maximum, "
            "etc.).\n"
            "3) Please focus only on {restriction} that would apply for "
            "{system_size_reminder}\n"
            "4) Pay close attention to clarifying details in parentheses, "
            "footnotes, or additional explanatory text.\n\n"
            "Example Inputs and Outputs:\n"
            "Text: 'For all WES there is a limitation of overall height "
            "of 200 feet (including blades).'\n"
            "Output: 200\n"
            "Text: 'The noise level of all SES shall be no greater than "
            "thirty-two (32) decibels measured from the nearest property "
            "line. This level may only be exceeded during short-term "
            "events such as utility outages and/or severe wind storms.'\n"
            "Output: 32\n"
            "Text: 'At no time shall a wind turbine tower, nacelle, or "
            "blade create shadow flicker on any non-participating "
            "landowner property'\n"
            "Output: 0\n"
            "Text: Solar Panels shall not exceed 22'6\" in height. The "
            "height is determined from the ground to the top of the panel "
            "at any angle.\n"
            "Output: 22.5\n"
        ),
    )

    G.add_edge("value", "units")
    G.add_node(
        "units",
        prompt=(
            "What are the units for the {restriction} for {tech}? Ensure "
            "that:\n"
            "1) You accurately identify the unit value associated with "
            "the restriction.\n"
            "2) The unit is expressed using standard, conventional unit "
            "names (e.g., 'feet', 'meters', 'acres', 'dBA', etc.). "
            "{unit_clarification}\n"
            "3) If multiple values are mentioned, return only the units "
            "for the most restrictive value that directly pertains to the "
            "restriction.\n"
            "\nExample Inputs and Outputs:\n"
            "Text: 'For all WES there is a limitation of overall height "
            "of 200 feet (including blades).'\n"
            "Output: 'feet'\n"
            "Text: 'The noise level of all SES shall be no greater than "
            "thirty-two (32) decibels measured from the nearest property "
            "line. This level may only be exceeded during short-term "
            "events such as utility outages and/or severe wind storms.'\n"
            "Output: 'dBA'\n"
            "Text: 'At no time shall a wind turbine tower, nacelle, or "
            "blade create shadow flicker on any non-participating "
            "landowner property'\n"
            "Output: 'hr/year'\n"
        ),
    )
    return G


def _add_prohibitions_extraction_nodes(G):  # noqa: N803
    """Add nodes and edges to extract 'prohibitions'"""

    G.add_edge("init", "is_proposed", condition=llm_response_starts_with_yes)
    G.add_node(
        "is_proposed",
        prompt=(
            "Is there reason to believe that this prohibition is only "
            "being proposed and not yet in effect? "
            "Please start your response with either 'Yes' or 'No' "
            "and briefly explain your answer."
        ),
    )
    G.add_edge(
        "is_proposed", "is_conditional", condition=llm_response_starts_with_no
    )
    G.add_node(
        "is_conditional",
        prompt=(
            "Does the prohibition, moratorium, or ban only apply "
            "conditionally? For example:\n"
            "  - Does it only apply to those who have not complied "
            "with the provisions in this text?\n"
            "  - Does it only apply within some distance of an area, "
            "landmark, or feature?\n"
            "  - Does it only apply to a subset of districts/areas "
            "within the jurisdiction?\n"
            "  - Does it only apply if a permit application has **not** "
            "been previously approved?\n"
            "  - Does it only apply if some other condition is met?\n"
            "  - etc.\n"
            "Please start your response with either 'Yes' or 'No' "
            "and briefly explain your answer."
        ),
    )
    G.add_edge(
        "is_conditional", "has_end_date", condition=llm_response_starts_with_no
    )
    G.add_node(
        "has_end_date",
        prompt=(
            "Does the legal text given an expiration date for the "
            "prohibition, moratorium, or ban? "
            "Please start your response with either 'Yes' or 'No' "
            "and briefly explain your answer."
        ),
    )
    G.add_edge("has_end_date", "final", condition=llm_response_starts_with_no)
    G.add_edge(
        "has_end_date",
        "check_end_date",
        condition=llm_response_starts_with_yes,
    )
    todays_date = datetime.now().strftime("%B %d, %Y")
    G.add_node(
        "check_end_date",
        prompt=(
            f"Today is {todays_date}. Has the prohibition, "
            "moratorium, or ban expired? "
            "Please start your response with either 'Yes' or 'No' "
            "and briefly explain your answer."
        ),
    )
    G.add_edge(
        "check_end_date", "final", condition=llm_response_starts_with_no
    )
    G.add_node(
        "final",
        prompt=(
            "Please respond based on our entire conversation so far. "
            "Return your answer as a dictionary in "
            "JSON format (not markdown). Your JSON file must include "
            "exactly two keys. The keys are 'summary' and 'section'. "
            "{SUMMARY_PROMPT} If the prohibition is a moratorium, be "
            "sure to include that distinction in your summary and "
            "provide any relevant expiration dates. {SECTION_PROMPT}"
        ),
    )
    return G


def setup_graph_permitted_use_districts(**kwargs):
    """Setup graph to extract permitted use districts for technology

    Parameters
    ----------
    **kwargs
        Keyword-value pairs to add to graph.

    Returns
    -------
    networkx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    feature_id = kwargs.get("feature_id", "")
    G = setup_graph_no_nodes(  # noqa: N806
        d_tree_name="Permitted use districts", **kwargs
    )

    G.add_node(
        "init",
        prompt=(
            "Does the following legal text explicitly define districts where "
            "{tech} (or similar) are {use_type}? {clarifications}"
            "Pay extra attention to titles and clarifying text found in "
            "parentheses and footnotes. Please start your response with "
            "either 'Yes' or 'No' and briefly explain your answer."
            '\n\n"""\n{text}\n"""'
        ),
    )
    G.add_edge(
        "init", "district_names", condition=llm_response_starts_with_yes
    )

    G.add_node(
        "district_names",
        prompt=(
            "What are all of the district names (and abbreviations if given) "
            "where {tech} (or similar) are {use_type}?"
        ),
    )

    if "primary" in feature_id:
        G.add_edge("district_names", "check_primary")
        G.add_node(
            "check_primary",
            prompt=(
                "Are these districts representative of locations where "
                "developers can site {tech} (or similar) as the primary "
                "use of the land/parcel/lot? Remember that this is true "
                "by assumption for all overlay districts. "
                "Please start your response with either 'Yes' or 'No' and "
                "briefly explain your answer."
            ),
        )
        G.add_edge(
            "check_primary", "final", condition=llm_response_starts_with_yes
        )
    elif "accessory" in feature_id:
        G.add_edge("district_names", "check_accessory")
        G.add_node(
            "check_accessory",
            prompt=(
                "Are these districts representative of locations where "
                "developers can site {tech} (or similar) as an accessory "
                "structure and/or as a secondary use of the land/parcel/lot? "
                "Please start your response with either 'Yes' or 'No' and "
                "briefly explain your answer."
            ),
        )
        G.add_edge(
            "check_accessory", "final", condition=llm_response_starts_with_yes
        )
    else:
        G.add_edge("district_names", "final")

    G.add_node(
        "final",
        prompt=(
            "Please respond based on our entire conversation so far. "
            "Return your answer as a dictionary in "
            "JSON format (not markdown). Your JSON file must include "
            "exactly three keys. The keys are 'value', 'summary', "
            "and 'section'. The value of the 'value' key "
            "should be a list of all district names (and abbreviations if "
            "given) where {tech} (or similar) "
            "are {use_type}, or `null` if the text does not "
            "mention this use type for {tech} (or similar). Use our "
            "conversation to fill out this value. {SUMMARY_PROMPT} "
            "{SECTION_PROMPT}"
        ),
    )
    return G


class BaseTextExtractor:
    """Base implementation for a text extractor"""

    SYSTEM_MESSAGE = (
        "You are a text extraction assistant. Your job is to extract only "
        "verbatim, **unmodified** excerpts from provided legal or policy "
        "documents. Do not interpret or paraphrase. Do not summarize. Only "
        "return exactly copied segments that match the specified scope. If "
        "the relevant content appears within a table, return the entire "
        "table, including headers and footers, exactly as formatted."
    )
    """System message for text extraction LLM calls"""
    _USAGE_LABEL = LLMUsageCategory.DOCUMENT_ORDINANCE_SUMMARY

    def __init__(self, llm_caller):
        """

        Parameters
        ----------
        llm_caller : LLMCaller
            LLM Caller instance used to extract ordinance info with.
        """
        self.llm_caller = llm_caller

    async def _process(self, text_chunks, instructions, is_valid_chunk):
        """Perform extraction processing"""
        logger.info(
            "Extracting summary text from %d text chunks asynchronously...",
            len(text_chunks),
        )
        logger.debug("Model instructions are:\n%s", instructions)
        outer_task_name = asyncio.current_task().get_name()
        summaries = [
            asyncio.create_task(
                self.llm_caller.call(
                    sys_msg=self.SYSTEM_MESSAGE,
                    content=f"{instructions}\n\n# TEXT #\n\n{chunk}",
                    usage_sub_label=self._USAGE_LABEL,
                ),
                name=outer_task_name,
            )
            for chunk in text_chunks
        ]
        summary_chunks = await asyncio.gather(*summaries)
        summary_chunks = [
            clean_backticks_from_llm_response(chunk)
            for chunk in summary_chunks
            if is_valid_chunk(chunk)
        ]

        text_summary = merge_overlapping_texts(summary_chunks)
        logger.debug(
            "Final summary contains %d tokens",
            ApiBase.count_tokens(
                text_summary,
                model=self.llm_caller.kwargs.get("model", "gpt-4"),
            ),
        )
        return text_summary
