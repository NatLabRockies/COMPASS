"""Ordinance Decision Tree Graph setup functions"""

from compass.extraction.common import (
    setup_graph_no_nodes,
    llm_response_starts_with_yes,
    llm_response_starts_with_no,
)


def setup_graph_wes_types(**kwargs):
    """Setup graph to get the largest turbine size in the ordinance text

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
            "Does the following text distinguish between multiple wind energy "
            "system sizes? Distinctions are often made as 'small', "
            "'personal', or 'private' vs 'large', 'commercial', or 'utility'. "
            "Sometimes the distinction uses actual MW values. "
            "Begin your response with either 'Yes' or 'No' and explain your "
            "answer."
            '\n\n"""\n{text}\n"""'
        ),
    )

    G.add_edge("init", "get_text", condition=llm_response_starts_with_yes)
    G.add_node(
        "get_text",
        prompt=(
            "What are the different wind energy system sizes this text "
            "mentions? List them in order of increasing size. Only include "
            "wind energy system types; do not include generic types or other "
            "energy system types."
        ),
    )
    G.add_edge("get_text", "final")
    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly two keys. The keys are "largest_wes_type" and '
            '"explanation". The value of the "largest_wes_type" key should '
            "be a string that labels the largest wind energy conversion "
            "system size mentioned in the text. The value of the "
            '"explanation" key should be a string containing a short '
            "explanation for your choice."
        ),
    )
    return G


def setup_multiplier(**kwargs):
    """Setup graph to extract a setbacks multiplier values for a feature

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
            "Does the text mention a multiplier that should be applied to a "
            "turbine dimension (e.g. height, rotor diameter, etc) to compute "
            "the setback distance from {feature}? "
            "Focus only on {feature}; do not respond based on any text "
            "related to {ignore_features}."
            "Remember that 1 is a valid multiplier, and treat any mention "
            "of 'fall zone' as a system height multiplier of 1. "
            "Please start your response with either 'Yes' or 'No' and "
            "explain your answer."
        ),
    )
    G.add_edge("init", "no_multiplier", condition=llm_response_starts_with_no)
    G.add_node(
        "no_multiplier",
        prompt=(
            "Does the ordinance give the setback from {feature} as a fixed "
            "distance value? Explain yourself."
        ),
    )
    G.add_edge(
        "no_multiplier", "units", condition=llm_response_starts_with_yes
    )
    G.add_edge(
        "no_multiplier", "out_static", condition=llm_response_starts_with_no
    )
    G.add_node(
        "units",
        prompt=(
            "What are the units for the setback from {feature}? "
            "Ensure that:\n1) You accurately identify the unit value "
            "associated with the setback.\n2) The unit is "
            "expressed using standard, conventional unit names (e.g., "
            '"feet", "meters", "miles" etc.)\n3) If multiple '
            "values are mentioned, return only the units for the most "
            "restrictive value that directly pertains to the setback.\n\n"
            "Example Inputs and Outputs:\n"
            'Text: "All Solar Farms shall be set back a distance of at least '
            'one thousand (1000) feet, from any primary structure"\n'
            'Output: "feet"\n'
        ),
    )
    G.add_edge("units", "out_static")
    G.add_node(
        "out_static",
        prompt=(
            "Please respond based on our entire conversation so far. "
            "Return your answer in JSON "
            "format (not markdown). Your JSON file must include exactly "
            'four keys. The keys are "value", "units", "summary", and '
            '"section". The value of the "value" key should be a '
            "**numerical** value corresponding to the setback distance value "
            "from {feature} or `null` if there was no such value. The value "
            'of the "units" key should be a string corresponding to the '
            "(standard) units of the setback distance value from {feature} "
            "or `null` if there was no such value. {SUMMARY_PROMPT} "
            "{SECTION_PROMPT}"
        ),
    )
    G.add_edge("init", "m_single", condition=llm_response_starts_with_yes)

    G.add_node(
        "m_single",
        prompt=(
            "Are multiple values given for the multiplier used to "
            "compute the setback distance value from {feature}? If so, "
            "select and state the largest one. Otherwise, repeat the single "
            "multiplier value that was given in the text. "
        ),
    )
    G.add_edge("m_single", "m_type")
    G.add_node(
        "m_type",
        prompt=(
            "What should the multiplier be applied to? Common acronyms "
            "include RD for rotor diameter and HH for hub height. Remember "
            "that system/total height is the tip-hight of the turbine. "
            "Select a value from the following list and explain yourself: "
            "['tip-height-multiplier', 'hub-height-multiplier', "
            "'rotor-diameter-multiplier]"
        ),
    )

    G.add_edge("m_type", "adder")
    G.add_node(
        "adder",
        prompt=(
            "Does the ordinance include a static distance value that "
            "should be added to the result of the multiplication? Do not "
            "confuse this value with static setback requirements. Ignore text "
            "with clauses such as 'no lesser than', 'no greater than', "
            "'the lesser of', or 'the greater of'. Begin your response with "
            "either 'Yes' or 'No' and explain your answer, stating the adder "
            "value if it exists."
        ),
    )
    G.add_edge("adder", "out_m", condition=llm_response_starts_with_no)
    G.add_edge("adder", "adder_eq", condition=llm_response_starts_with_yes)

    G.add_node(
        "adder_eq",
        prompt=(
            "Does the adder value you identified satisfy the following "
            "equation: 'multiplier * height + <adder>'? Please begin your "
            "response with either 'Yes' or 'No' and explain your answer."
        ),
    )
    G.add_edge("adder_eq", "out_m", condition=llm_response_starts_with_no)
    G.add_edge(
        "adder_eq",
        "conversion",
        condition=llm_response_starts_with_yes,
    )
    G.add_node(
        "conversion",
        prompt=(
            "If the adder value is not given in feet, convert "
            "it to feet (remember that there are 3.28084 feet in one meter "
            "and 5280 feet in one mile). Show your work step-by-step "
            "if you had to perform a conversion."
        ),
    )
    G.add_edge("conversion", "out_m")

    G.add_node(
        "out_m",
        prompt=(
            "Please respond based on our entire conversation so far. "
            "Return your answer in JSON "
            "format (not markdown). Your JSON file must include exactly five "
            'keys. The keys are "mult_value", "mult_type", "adder", '
            '"summary", and "section". The value of the '
            '"mult_value" key should be a **numerical** value corresponding '
            "to the multiplier value we determined earlier. The value of the "
            '"mult_type" key should be a string corresponding to the '
            "dimension that the multiplier should be applied to, as we "
            'determined earlier. The value of the "adder" key should be a '
            "**numerical** value corresponding to the static value to be "
            "added to the total setback distance after multiplication, as we "
            "determined earlier, or `null` if there is no such value. "
            "{SUMMARY_PROMPT} {SECTION_PROMPT}"
        ),
    )

    return G


def setup_conditional(**kwargs):
    """Setup graph to extract min/max setback values for a feature

    Min/Max setback values (after application of multiplier) are
    typically given within the context of 'the greater of' or
    'the lesser of' clauses.

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
            "Focus only on setback from {feature}; do not respond based "
            "on any text related to {ignore_features}."
            "Does the setback from {feature} mention a minimum or maximum "
            "static setback distance **regardless of the outcome** of the "
            "multiplier calculation? This is often phrased as 'the greater "
            "of' or 'the lesser of'. Do not confuse this value with static "
            "values to be added to multiplicative setbacks. Begin your "
            "response with either 'Yes' or 'No' and explain your answer."
        ),
    )

    G.add_edge("init", "conversions", condition=llm_response_starts_with_yes)
    G.add_node(
        "conversions",
        prompt=(
            "Tell me the minimum and/or maximum setback distances, "
            "converting to feet if necessary (remember that there are "
            "3.28084 feet in one meter and 5280 feet in one mile). "
            "Explain your answer and show your work if you had to perform "
            "a conversion."
        ),
    )

    G.add_edge("conversions", "out_condition")
    G.add_node(
        "out_condition",
        prompt=(
            "Please respond based on our entire conversation so far. "
            "Return your answer in JSON "
            "format (not markdown). Your JSON file must include exactly two "
            'keys. The keys are "min_dist" and "max_dist". The value of the '
            '"min_dist" key should be a **numerical** value corresponding to '
            "the minimum setback value from {feature} we determined earlier, "
            'or `null` if no such value exists. The value of the "max_dist" '
            "key should be a **numerical** value corresponding to the maximum "
            "setback value from {feature}  we determined earlier, or `null` "
            "if no such value exists."
        ),
    )

    return G
