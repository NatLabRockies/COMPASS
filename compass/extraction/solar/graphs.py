"""Solar ordinance decision tree graph setup functions"""

from compass.common import (
    setup_graph_no_nodes,
    llm_response_starts_with_yes,
    llm_response_starts_with_no,
)


def setup_graph_sef_types(**kwargs):
    """Setup graph to get the largest solar farm size in the text

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
            "Does the following text distinguish between multiple solar "
            "energy farm sizes? Distinctions are often made as 'small', "
            "'personal', or 'private' vs 'large', 'commercial', or 'utility'. "
            "Sometimes the distinction uses actual MW values. "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain your answer."
            '\n\n"""\n{text}\n"""'
        ),
    )

    G.add_edge("init", "get_text", condition=llm_response_starts_with_yes)
    G.add_node(
        "get_text",
        prompt=(
            "What are the different solar energy farm sizes regulated by "
            "this ordinance? List them in order of increasing size. "
            "Include any relevant numerical qualifiers in the name, if "
            "appropriate. Only include solar energy farm types; do not "
            "include generic types or other energy system types."
        ),
    )
    G.add_edge("get_text", "final")
    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer as a dictionary in JSON format (not markdown). Your "
            "JSON file must include exactly two keys. The keys are "
            "'largest_sef_type' and 'explanation'. The value of the "
            "'largest_sef_type' key should be a string that labels the "
            "largest solar energy system size regulated by this ordinance. "
            "The value of the 'explanation' key should be a string containing "
            "a short explanation for your choice."
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
            "Does the text mention a multiplier that should be applied to the "
            "structure height to compute the setback distance from {feature} "
            "for {tech}? "
            "Focus only on {feature}; do not respond based on any text "
            "related to {ignore_features}. "
            "Please only consider setbacks specifically for systems that "
            "would typically be defined as {tech} based on the text itself "
            "— for example, systems intended for electricity generation or "
            "sale, or those above thresholds such as height or rated "
            "capacity. Ignore any requirements that apply only to smaller "
            "or clearly non-commercial systems. "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain your answer."
        ),
    )
    G.add_edge("init", "no_multiplier", condition=llm_response_starts_with_no)
    G.add_node(
        "no_multiplier",
        prompt=(
            "Does the ordinance give the setback from {feature} as a fixed "
            "distance value? "
            "Focus only on {feature}; do not respond based on any text "
            "related to {ignore_features}. "
            "Please only consider setbacks specifically for systems that "
            "would typically be defined as {tech} based on the text itself "
            "— for example, systems intended for electricity generation or "
            "sale, or those above thresholds such as height or rated "
            "capacity. Ignore any requirements that apply only to smaller "
            "or clearly non-commercial systems. "
            "Please start your response with either 'Yes' or "
            "'No' and briefly explain your answer."
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
            "'feet', 'meters', 'miles' etc.)\n3) If multiple "
            "values are mentioned, return only the units for the most "
            "restrictive value that directly pertains to the setback.\n\n"
            "Example Inputs and Outputs:\n"
            "Text: 'All WECS Towers shall be set back a distance of at least "
            "one thousand (1000) feet, from any primary structure'\n"
            "Output: 'feet'\n"
        ),
    )
    G.add_edge("units", "out_static")
    G.add_node(
        "out_static",
        prompt=(
            "Please respond based on our entire conversation so far. "
            "Return your answer in JSON "
            "format (not markdown). Your JSON file must include exactly "
            "four keys. The keys are 'value', 'units', 'summary', and "
            "'section'. The value of the 'value' key should be a "
            "**numerical** value corresponding to the setback distance value "
            "from {feature} or `null` if there was no such value. The value "
            "of the 'units' key should be a string corresponding to the "
            "(standard) units of the setback distance value from {feature} "
            "or `null` if there was no such value. "
            "As before, focus only on setbacks specifically for systems that "
            "would typically be defined as {tech} based on the text itself. "
            "{SUMMARY_PROMPT} {SECTION_PROMPT}"
        ),
    )
    G.add_edge("init", "m_single", condition=llm_response_starts_with_yes)

    G.add_node(
        "m_single",
        prompt=(
            "Are multiple values given for the multiplier used to "
            "compute the setback distance value from {feature} for {tech}? "
            "Remember to ignore any text related to {ignore_features}. "
            "Focus only on setbacks specifically for systems that would "
            "typically be defined as {tech} based on the text itself — for "
            "example, systems intended for electricity generation or sale, "
            "or those above thresholds such as height or rated capacity. "
            "Ignore any requirements that apply only to smaller or clearly "
            "non-commercial systems. "
            "If so, select and state the largest one. Otherwise, repeat the "
            "single multiplier value that was given in the text. "
        ),
    )
    G.add_edge("m_single", "adder")
    G.add_node(
        "adder",
        prompt=(
            "Does the ordinance for the setback from {feature} include a "
            "static distance value that should be added to the result of "
            "the multiplication? "
            "Remember to ignore any text related to {ignore_features}. "
            "Focus only on setbacks specifically for systems that would "
            "typically be defined as {tech} based on the text itself — for "
            "example, systems intended for electricity generation or sale, "
            "or those above thresholds such as height or rated capacity. "
            "Ignore any requirements that apply only to smaller or clearly "
            "non-commercial systems. "
            "Do not confuse this value with static setback requirements. "
            "Ignore text with clauses such as 'no lesser than', 'no greater "
            "than', 'the lesser of', or 'the greater of'. Please start your "
            "response with either 'Yes' or 'No' and briefly explain your "
            "answer, stating the adder value if it exists."
        ),
    )
    G.add_edge("adder", "out_no_adder", condition=llm_response_starts_with_no)
    G.add_edge("adder", "adder_eq", condition=llm_response_starts_with_yes)

    G.add_node(
        "adder_eq",
        prompt=(
            "Does the adder value you identified satisfy the following "
            "equation: `multiplier * height + <adder>`? Begin your "
            "response with either 'Yes' or 'No' and briefly explain your "
            "answer."
        ),
    )
    G.add_edge(
        "adder_eq", "out_no_adder", condition=llm_response_starts_with_no
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
            "Return your answer as a single dictionary in JSON "
            "format (not markdown). Your JSON file must include exactly four "
            "keys. The keys are 'mult_value', 'adder', 'summary', and "
            "'section'. The value of the 'mult_value' key should be a "
            "**numerical** value corresponding to the multiplier value we "
            "determined earlier. The value of the 'adder' key should be a "
            "**numerical** value corresponding to the static value to be "
            "added to the total setback distance after multiplication, as we "
            "determined earlier, or `null` if there is no such value. "
            "{SUMMARY_PROMPT} {SECTION_PROMPT}"
        ),
    )
    G.add_node(
        "out_no_adder",
        prompt=(
            "Please respond based on our entire conversation so far. "
            "Return your answer as a single dictionary in JSON "
            "format (not markdown). Your JSON file must include exactly three "
            "keys. The keys are 'mult_value', 'summary', and 'section'. The "
            "value of the 'mult_value' key should be a **numerical** value "
            "corresponding to the multiplier value we determined earlier. "
            "{SUMMARY_PROMPT} {SECTION_PROMPT}"
        ),
    )

    return G
