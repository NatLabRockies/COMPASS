"""Document validation decision tree graph setup functions"""

from compass.common import (
    setup_graph_no_nodes,
    llm_response_starts_with_yes,
    llm_response_starts_with_no,
)


def setup_graph_correct_document_type(**kwargs):
    """Setup graph to check for correct document type in legal text

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
            "Does the following text excerpt detail in-effect legal statutes? "
            "Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
            '\n\n"""\n{text}\n"""'
        ),
    )

    G.add_edge("init", "is_model", condition=llm_response_starts_with_yes)
    G.add_node(
        "is_model",
        prompt=(
            "Is the text an excerpt from a model ordinance or other "
            "kind of model law? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain why you chose your answer."
        ),
    )

    G.add_edge("is_model", "is_pd", condition=llm_response_starts_with_no)
    G.add_node(
        "is_pd",
        prompt=(
            "Is the text an excerpt from a planning document? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain why you chose your answer."
        ),
    )

    G.add_edge("is_pd", "is_pres", condition=llm_response_starts_with_no)
    G.add_node(
        "is_pres",
        prompt=(
            "Is the text an excerpt from a presentation? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain why you chose your answer."
        ),
    )

    G.add_edge("is_pres", "is_draft", condition=llm_response_starts_with_no)
    G.add_node(
        "is_draft",
        prompt=(
            "Is the text an excerpt from a draft document? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain why you chose your answer."
        ),
    )

    G.add_edge("is_draft", "is_report", condition=llm_response_starts_with_no)
    G.add_node(
        "is_report",
        prompt=(
            "Is the text an excerpt from a report or summary "
            "document? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain why you chose your answer."
        ),
    )

    G.add_edge(
        "is_report", "is_article", condition=llm_response_starts_with_no
    )
    G.add_node(
        "is_article",
        prompt=(
            "Is the text an excerpt from a news article or "
            "other media? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain why you chose your answer."
        ),
    )

    G.add_edge("is_article", "final", condition=llm_response_starts_with_no)
    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer as a dictionary in JSON format (not markdown). Your JSON "
            "file must include exactly three keys:\n\n"
            "1. **'summary'** (string) - A concise summary of the text.\n"
            "2. **'type'** (string) - The best-fitting category for the "
            "source of the text.\n"
            "3. **'{key}'** (boolean) -\n"
            "\t- `true` if the text is a **legally binding regulation**.\n"
            "\t- `false` if the text belongs to any other type of document or "
            "if you cannot tell for certain one way or another.\n\n"
        ),
    )
    return G


def setup_graph_correct_jurisdiction_type(jurisdiction, **kwargs):
    """Setup graph to check for correct jurisdiction type in legal text

    Parameters
    ----------
    jurisdiction : compass.utilities.location.Jurisdiction
        Jurisdiction for which validation is being performed.
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
            "Does the following legal text explicitly include enough "
            "information to reasonably conclude what type of "
            "jurisdiction it applies to? Common types of jurisdictions "
            "include 'state', 'county', 'city', 'township',' borough', etc. "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain your answer."
            '\n\n"""\n{text}\n"""'
        ),
    )

    G.add_edge("init", "is_state", condition=llm_response_starts_with_yes)
    G.add_node(
        "is_state",
        prompt=(
            "Based on the legal text, is it reasonable to conclude that the "
            "provisions within apply to the entire state of "
            f"{jurisdiction.state}, either directly or through reference to "
            "a statewide statute, agency, or regulatory authority? If the "
            "text only applies to a county, municipality, or other local "
            f"subdivision within {jurisdiction.state}, or if there is no "
            "reasonable basis to infer statewide application, respond with "
            "'No'. Start your response with 'Yes' or 'No' and briefly explain."
        ),
    )
    node_to_connect = "is_state"

    if jurisdiction.county:
        G.add_edge(
            node_to_connect, "is_county", condition=llm_response_starts_with_no
        )
        G.add_edge(
            node_to_connect, "final", condition=llm_response_starts_with_yes
        )
        G.add_node(
            "is_county",
            prompt=(
                "Based on the legal text, is it reasonable to conclude that "
                "the provisions within apply to the entire area governed by "
                f"{jurisdiction.full_county_phrase} (including unincorporated "
                "areas), either directly or through reference to a "
                "county-wide code, planning authority, commission, or joint "
                "resolution with other local governments? If the provisions "
                "within the text apply **only** to a subdivision of "
                f"{jurisdiction.full_county_phrase} (such as a city or "
                "township), or the scope is unclear, respond with 'No'. "
                "Start your answer with 'Yes' or 'No' and explain briefly."
            ),
        )
        G.add_edge(
            "is_county", "final", condition=llm_response_starts_with_yes
        )
        node_to_connect = "is_county"

    if jurisdiction.subdivision_name:
        G.add_edge(
            node_to_connect, "is_city", condition=llm_response_starts_with_no
        )
        G.add_edge(
            node_to_connect, "final", condition=llm_response_starts_with_yes
        )
        G.add_node(
            "is_city",
            prompt=(
                "Based on the legal text, is it reasonable to conclude that "
                "the provisions apply specifically to "
                f"{jurisdiction.full_subdivision_phrase} (rather than a "
                "county, state, or federal jurisdiction)? If the text instead "
                "applies to a broader jurisdiction, or does not provide a "
                "reasonable basis to infer that it is limited to municipal "
                "governance, respond with 'No'. "
                "Start your response with 'Yes' or 'No' and briefly explain."
            ),
        )
        node_to_connect = "is_city"

    G.add_edge(node_to_connect, "final")
    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer as a dictionary in JSON format (not markdown). Your JSON "
            "file must include exactly two keys. The keys are "
            "'correct_jurisdiction' and 'explanation'. The value of the "
            "'correct_jurisdiction' key should be a boolean that is set to "
            "`true` **only if** it is reasonable to conclude that the "
            "provisions within apply to the entire area (i.e. "
            f"{jurisdiction.type}-wide) governed by {jurisdiction.full_name} "
            "(`false` otherwise). The value of the 'explanation' key should "
            "be a string containing a brief explanation for your choice. "
        ),
    )
    return G


def setup_graph_correct_jurisdiction_from_url(jurisdiction, **kwargs):
    """Setup graph to check for correct jurisdiction in URL

    Parameters
    ----------
    jurisdiction : compass.utilities.location.Jurisdiction
        Jurisdiction for which validation is being performed.
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
            f"Does the following URL explicitly mention {jurisdiction.state} "
            "state in some way (e.g. either by full name or abbreviation)? "
            "**Do not** answer based on auxiliary information like county or "
            "city names. "
            "Please start your response with either 'Yes' or 'No' and explain "
            "your answer."
            "\n\nURL: '{url}\n'"
        ),
    )

    node_to_connect = "init"
    keys_to_collect = {"correct_state": f"{jurisdiction.state} state"}

    if jurisdiction.county:
        G.add_edge(
            node_to_connect,
            "mentions_county",
            condition=llm_response_starts_with_yes,
        )
        G.add_node(
            "mentions_county",
            prompt=(
                "Does the URL explicitly mention "
                f"{jurisdiction.full_county_phrase} in some way (e.g. either "
                "by full name or abbreviation)? **Do not** answer based on "
                "auxiliary information like state or city names. "
                "Please start your response with either 'Yes' or 'No' and "
                "briefly explain your answer."
                "\n\nURL: '{url}\n'"
            ),
        )
        keys_to_collect["correct_county"] = (
            f"{jurisdiction.full_county_phrase}"
        )
        node_to_connect = "mentions_county"

    if jurisdiction.subdivision_name:
        G.add_edge(
            node_to_connect,
            "mentions_city",
            condition=llm_response_starts_with_yes,
        )
        G.add_node(
            "mentions_city",
            prompt=(
                "Does the URL explicitly mention "
                f"{jurisdiction.full_subdivision_phrase} in some way (e.g. "
                "either by full name or abbreviation)? **Do not** answer "
                "based on auxiliary information like state or county names. "
                "Please start your response with either 'Yes' or 'No' and "
                "briefly explain your answer."
                "\n\nURL: '{url}\n'"
            ),
        )
        keys_to_collect[f"correct_{jurisdiction.type.casefold()}"] = (
            f"{jurisdiction.full_subdivision_phrase}"
        )
        node_to_connect = "mentions_city"

    G.add_edge(
        node_to_connect, "final", condition=llm_response_starts_with_yes
    )
    G.add_node("final", prompt=_compile_final_url_prompt(keys_to_collect))
    return G


def _compile_final_url_prompt(keys_to_collect):
    """Compile final URL instruction prompt"""
    num_keys = len(keys_to_collect) + 1
    num_keys = f"Your JSON file must include exactly {num_keys} keys. "

    out_keys = ", ".join([f"'{key}'" for key in keys_to_collect])
    out_keys = f"The keys are {out_keys} and 'explanation'. "

    explain_text = _compile_url_key_explain_text(keys_to_collect)

    return (
        "Respond based on our entire conversation so far. Return your "
        "answer as a dictionary in JSON format (not markdown). "
        f"{num_keys}{out_keys}{explain_text}"
    )


def _compile_url_key_explain_text(keys_to_collect):
    """Compile explanations ofr each output key"""
    explain_text = []
    for key, name in keys_to_collect.items():
        explain_text.append(
            f"The value of the '{key}' key should be a boolean that is set to "
            f"`True` if the URL explicitly mentions {name} in some way "
            "(`False` otherwise). "
        )

    choices = "choices" if len(keys_to_collect) > 1 else "choice"
    explain_text.append(
        "The value of the 'explanation' key should be a string containing a "
        f"short explanation for your {choices}. "
    )
    return "".join(explain_text)
