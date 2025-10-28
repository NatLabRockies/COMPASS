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
    networkx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    G = setup_graph_no_nodes(  # noqa: N806
        d_tree_name="Solar Energy Farm types", **kwargs
    )

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
            "What are the different solar energy farm sizes **regulated by "
            "this ordinance**? List them in order of increasing size. "
            "Include any relevant numerical qualifiers in the name, if "
            "appropriate. Only include systems that resemble ground-mounted "
            "solar energy farms; do not include other solar energy system "
            "types lice CSP or roof-mounted systems or other technologies "
            "like wind energy systems, geothermal energy systems, etc."
        ),
    )
    G.add_edge("get_text", "get_regulated")
    G.add_node(
        "get_regulated",
        prompt=(
            "Are any of these systems **not** regulated by this ordinance?"
        ),
    )

    G.add_edge("get_regulated", "get_largest")
    G.add_node(
        "get_largest",
        prompt=(
            "What is the **largest** solar energy farm size that **is "
            "regulated by this ordinance**?"
        ),
    )

    G.add_edge("get_largest", "check_matches_definition")
    G.add_node(
        "check_matches_definition",
        prompt=(
            "Does the ordinance explicitly define this system as large, "
            "commercial, utility-scale, or something akin to that? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain your answer."
        ),
    )

    G.add_edge(
        "check_matches_definition",
        "final_large",
        condition=llm_response_starts_with_yes,
    )
    G.add_edge(
        "check_matches_definition",
        "check_scale_reason",
        condition=llm_response_starts_with_no,
    )
    G.add_node(
        "check_scale_reason",
        prompt=(
            "Would a reasonable person classify this kind of system as a "
            "**large**, commercial, or even utility-scale** solar energy farm "
            "(e.g. with the primary purpose of generating electricity for "
            "sale, as opposed to small, residential, roof-mounted, private, "
            "or other kinds of 'small' systems)? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain your answer."
        ),
    )

    G.add_edge(
        "check_scale_reason",
        "final_large",
        condition=llm_response_starts_with_yes,
    )
    G.add_edge(
        "check_scale_reason",
        "final_small",
        condition=llm_response_starts_with_no,
    )
    G.add_node(
        "final_large",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer as a dictionary in JSON format (not markdown). Your "
            "JSON file must include exactly two keys. The keys are "
            "'largest_sef_type' and 'explanation', and 'is_large'. The value "
            "of the 'largest_sef_type' key should be a string that labels the "
            "largest solar energy system size **regulated by this "
            "ordinance**. The value of the 'explanation' key should be a "
            "string containing a short explanation for your choice. The value "
            "of the 'is_large' key should be the boolean value `true`, since "
            "we determined this is a large-scale system."
        ),
    )
    G.add_node(
        "final_small",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer as a dictionary in JSON format (not markdown). Your "
            "JSON file must include exactly two keys. The keys are "
            "'largest_sef_type' and 'explanation', and 'is_large'. The value "
            "of the 'largest_sef_type' key should be a string that labels the "
            "largest solar energy system size **regulated by this "
            "ordinance**. The value of the 'explanation' key should be a "
            "string containing a short explanation for your choice. The value "
            "of the 'is_large' key should be the boolean value `false`, since "
            "we determined this is not a large-scale system."
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
    networkx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    G = setup_graph_no_nodes(  # noqa: N806
        d_tree_name="Setback distance", **kwargs
    )

    G.add_node(
        "init",
        prompt=(
            "Does the text mention a multiplier that should be applied to the "
            "structure height to compute the setback distance from {feature} "
            "for {tech}? "
            "Please consider only {feature}; do not respond based on any text "
            "related to {ignore_features}. "
            "Please also only consider setbacks specifically for "
            "{system_size_reminder}"
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
            "Please consider only on {feature}; do not respond based on any "
            "text related to {ignore_features}. "
            "Please also only consider setbacks specifically for "
            "{system_size_reminder}"
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
            "As before, focus only on setbacks that would apply for"
            "{system_size_reminder}"
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
            "Please only consider setbacks specifically for "
            "{system_size_reminder}"
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
            "Please only consider setbacks specifically for "
            "{system_size_reminder}"
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
