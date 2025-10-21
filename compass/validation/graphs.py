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
    networkx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    doc_is_from_ocr = kwargs.pop("doc_is_from_ocr", False)

    G = setup_graph_no_nodes(  # noqa: N806
        d_tree_name="Correct document type", **kwargs
    )
    G.add_node(
        "init",
        prompt=(
            "Does the following text resemble an excerpt from a legal "
            "statute, such as an ordinance or code? "
            "Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
            '\n\n"""\n{text}\n"""'
        ),
    )

    G.add_edge("init", "check_for_laws", condition=llm_response_starts_with_no)
    G.add_node(
        "check_for_laws",
        prompt=(
            "Does the text excerpt detail legal statutes/regulations? "
            "Please start your response with either 'Yes' or 'No' and "
            "briefly explain your answer."
        ),
    )

    G.add_edge("init", "is_model", condition=llm_response_starts_with_yes)
    G.add_edge(
        "check_for_laws", "is_model", condition=llm_response_starts_with_yes
    )
    G.add_node(
        "is_model",
        prompt=(
            "Does this text appear to be from a model ordinance or other "
            "kind of model law? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain why you chose your answer."
        ),
    )

    G.add_edge("is_model", "in_effect", condition=llm_response_starts_with_no)
    G.add_node(
        "in_effect",
        prompt=(
            "Is this regulation in effect? Use only the **document's "
            "content** to determine your answer (ignore editing/version "
            "labels, track changes, and metadata).\n\n"
            "Decision rules:\n\n"
            "* If {tech} regulations are present, rely **only on the status "
            "of those regulations in particular**.\n"
            "* If the text explicitly states adoption/approval/enactment "
            "status, rely only on that.\n"
            "* If the text contains proposal-stage indicators (e.g., "
            '"proposed ordinance," "notice of proposed rulemaking," "for '
            'public comment," "draft for review," "public hearing scheduled," '
            '"introduced," "pending adoption," etc.), treat it as **not** in '
            "effect.\n"
            '* If the text contains adoption indicators (e.g., "adopted," '
            '"enacted," "approved," "codified," "final rule," "effective '
            '[date]"), treat it as in effect.\n'
            "* If the text contains other final-stage indicators (e.g., "
            '"rejected," "not approved," etc.), treat it as **not** in '
            "effect.\n\n"
            "**IMPORTANT**\n"
            "If evidence is mixed, or the text does not explicitly give the "
            "adoption status, or there is not enough information to "
            "**confidently** conclude one way or another, default to "
            '"Yes".\n\n'
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain why you chose your answer."
        ),
    )

    G.add_edge(
        "in_effect", "is_meeting", condition=llm_response_starts_with_yes
    )
    G.add_node(
        "is_meeting",
        prompt=(
            "Does this text appear to be from town or board meeting notes? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain why you chose your answer."
        ),
    )

    G.add_edge(
        "is_meeting", "is_public_notice", condition=llm_response_starts_with_no
    )
    G.add_node(
        "is_public_notice",
        prompt=(
            "Does this text appear to be from a public notice or letter? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain why you chose your answer."
        ),
    )

    G.add_edge(
        "is_public_notice",
        "is_single_project",
        condition=llm_response_starts_with_no,
    )
    G.add_node(
        "is_single_project",
        prompt=(
            "Does this text appear to apply for a single specific project? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain why you chose your answer."
        ),
    )

    G.add_edge(
        "is_single_project",
        "is_planning_doc",
        condition=llm_response_starts_with_no,
    )
    G.add_node(
        "is_planning_doc",
        prompt=(
            "Does this text appear to be from a project planning document? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain why you chose your answer."
        ),
    )

    G.add_edge(
        "is_planning_doc", "is_pres", condition=llm_response_starts_with_no
    )
    G.add_node(
        "is_pres",
        prompt=(
            "Does this text appear to be from a presentation? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain why you chose your answer."
        ),
    )

    G.add_edge("is_pres", "is_draft", condition=llm_response_starts_with_no)

    is_draft_prompt = [
        "Does this text appear to be from a document that is currently "
        "being edited or formatted, such as a draft or work in progress?\n"
        "\n**Important**:\n"
    ]
    if doc_is_from_ocr:
        is_draft_prompt.append(
            "* Disregard formatting inconsistencies, typographical errors, or "
            "visual artifacts (such as OCR noise, broken lines, or unusual "
            "spacing). These do **not** indicate draft status unless "
            "supported by actual content-based cues."
        )
    is_draft_prompt.append(
        "* Do **not** assume that a document is a draft simply because it "
        "refers to amendments, revisions of law, or changing legal "
        "standards. Many finalized legal documents contain such "
        "references as part of their normal content.\n"
        "* Do **not** assume the document is a draft for these common "
        'non-indicative phrases: references to amendments, "shall", "may", '
        '"as amended", "upon adoption", "effective date", "adopted", or the '
        'presence of dates (including "Adopted Date", "Effective Date", '
        '"Final Adopted"). These appear in both final and draft legal texts '
        'and should **not** by themselves trigger a "Yes".\n'
        "* Do **not** assume that a document is a draft if it contains "
        'ambiguous phrases like "Section for Revision" but **no** '
        "placeholders, comments, TODOs, or other explicit editing markers.\n"
        "* Do **not** treat blank fields, bracketed fill-in areas, or form "
        "templates appearing in **appendices, attachments, or exhibits** "
        "(e.g., 'Appendix,' 'Form,' or 'Application Template') as indicators "
        "of an unfinished draft. Many finalized ordinances and regulations "
        "include such templates for public or administrative use.\n"
        "\nFocus instead on signs of incompleteness or active "
        "editing, such as (but not limited to):\n"
        '* explicit labels: "DRAFT", "DRAFT VERSION", "NOT FINAL", "FOR '
        'REVIEW", "WORKING VERSION", "DO NOT PUBLISH".\n'
        '* placeholders: "TBD", "TBA", "INSERT TEXT", "INSERT [SECTION]", '
        '"INSERT HERE", "___", "xxx", "[insert...]", "[?]", "[TO BE '
        'DETERMINED]".\n'
        "* editorial comments or markup: bracketed comments like "
        '"[Comment: ...]", "/* ... */", HTML/XML comments `<!-- -->`, '
        'tracked changes markers "Track Changes", "redline", "stet", change '
        "bars, or visible revision marks.\n"
        '* explicit TODOs or instructions to editors: "TODO", "REPLACE", '
        '"REVISE SECTION", "CHECK CITATION".'
        '* clear formatting markers left for future editing: "## HEADLINE ##, '
        "repeated underscores, sequences of asterisks used as placeholders, "
        'or visible template text such as "Section for Revision" immediately '
        "adjacent to other placeholders or TODOs.\n\n"
        "When both finalized indicators (e.g., enactment dates, official "
        "signatures, or filing statements) and bracketed placeholders "
        "appear, prioritize the **finalization evidence** unless there are "
        "explicit editing markers.\n"
        "Please begin your answer with **Yes** or **No**, and briefly "
        "explain your reasoning based only on these content-based signals."
    )
    is_draft_prompt = "\n".join(is_draft_prompt)
    G.add_node("is_draft", prompt=is_draft_prompt)

    G.add_edge("is_draft", "is_report", condition=llm_response_starts_with_no)
    G.add_node(
        "is_report",
        prompt=(
            "Does this text appear to be from a report or summary "
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
            "Does this text appear to be from a news article or "
            "other media? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain why you chose your answer."
        ),
    )

    G.add_edge(
        "is_article", "is_court_doc", condition=llm_response_starts_with_no
    )
    G.add_node(
        "is_court_doc",
        prompt=(
            "Does this text appear to be from a lawsuit, legal complaint, "
            "application form, or other legal or court document that is not "
            "intended to detail specific laws, ordinances, and/or "
            "regulations? "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain why you chose your answer."
        ),
    )

    G.add_edge("is_court_doc", "final", condition=llm_response_starts_with_no)
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
    jurisdiction : Jurisdiction
        Jurisdiction for which validation is being performed.
    **kwargs
        Keyword-value pairs to add to graph.

    Returns
    -------
    networkx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    G = setup_graph_no_nodes(  # noqa: N806
        d_tree_name="Correct jurisdiction type", **kwargs
    )

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

    names_we_want = _jurisdiction_names_to_extract(jurisdiction)

    G.add_edge("init", "has_name", condition=llm_response_starts_with_yes)
    G.add_node(
        "has_name",
        prompt=(
            "Does the legal text explicitly include enough information to "
            "reasonably determine the **full name** of the jurisdiction it "
            f"applies to? We want to know at least {names_we_want}. "
            "Please start your response with either 'Yes' or 'No' and briefly "
            "explain your answer."
        ),
    )

    G.add_edge("has_name", "is_state", condition=llm_response_starts_with_yes)
    G.add_node(
        "is_state",
        prompt=(
            "Based on the legal text, is it reasonable to conclude that the "
            "provisions within apply specifically to the entire state of "
            f"**{jurisdiction.state}**, either directly or through reference "
            "to a statewide statute, agency, or regulatory authority? If the "
            "text only applies to a county, municipality, or other local "
            f"subdivision within {jurisdiction.state}, or if the text applies "
            "to a different state entirely, or if there is no "
            "reasonable basis to infer statewide application, respond with "
            "'No'. Start your response with 'Yes' or 'No' and briefly explain."
        ),
    )

    if not jurisdiction.county and not jurisdiction.subdivision_name:
        G.add_edge(
            "is_state",
            "has_state_name",
            condition=llm_response_starts_with_yes,
        )
        G.add_edge("is_state", "final", condition=llm_response_starts_with_no)
        G.add_node(
            "has_state_name",
            prompt=(
                "Based on the legal text, is there clear and specific "
                "evidence that the ordinance applies specifically to "
                f"**{jurisdiction.full_name_the_prefixed}**? This could "
                f"include a direct mention of **{jurisdiction.state}**, a "
                "title, heading, or citation indicating it's an ordinance for "
                f"{jurisdiction.state} state, or other language that "
                f"reasonably ties the text to {jurisdiction.full_name} "
                "specifically. Generic references such as 'the state' or "
                "'State Zoning Administrator' are not sufficient on their own "
                "unless clearly linked to "
                f"{jurisdiction.full_name_the_prefixed}. "
                "Start your response with 'Yes' or 'No' and explain briefly."
            ),
        )
        G.add_edge("has_state_name", "final")

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
                "the provisions within apply specifically to "
                f"**{jurisdiction.full_county_phrase}** "
                "(incorporated or unincorporated areas), either directly or "
                "through reference to a county-wide code, planning authority, "
                "commission, or joint resolution with other local "
                "governments? If the provisions within the text apply "
                "**only** to a **subdivision** of "
                f"{jurisdiction.full_county_phrase} (such as a city or "
                "township), or the text applies to a different county or "
                "borough entirely, or if the scope is unclear, respond with "
                "'No'. "
                "Start your answer with 'Yes' or 'No' and explain briefly."
            ),
        )
        if not jurisdiction.subdivision_name:
            G.add_edge(
                "is_county", "final", condition=llm_response_starts_with_no
            )
            G.add_edge(
                "is_county",
                "has_county_name",
                condition=llm_response_starts_with_yes,
            )
            G.add_node(
                "has_county_name",
                prompt=(
                    "Based on the legal text, is there clear and specific "
                    "evidence that the ordinance applies specifically to "
                    f"**{jurisdiction.full_name_the_prefixed}**? This could "
                    f"include a direct mention of **{jurisdiction.county}**, "
                    "a title, heading, or citation indicating it's an "
                    f"ordinance for {jurisdiction.county} "
                    f"{jurisdiction.type.casefold()}, or other language that "
                    f"reasonably ties the text to {jurisdiction.full_name} "
                    "specifically. Generic references such as 'the "
                    f"{jurisdiction.type.casefold()}' or "
                    f"'{jurisdiction.type} Zoning Administrator' are not "
                    "sufficient on their own unless clearly linked to "
                    f"{jurisdiction.full_name_the_prefixed}. "
                    "Start your response with 'Yes' or 'No' and explain "
                    "briefly."
                ),
            )
            G.add_edge("has_county_name", "final")
        else:
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
                f"**{jurisdiction.full_subdivision_phrase_the_prefixed}** "
                "(rather than a county, state, federal jurisdiction, or a "
                f"different {jurisdiction.type.casefold()})? If the text "
                "instead applies to a broader jurisdiction, or applies to "
                f"a different {jurisdiction.type.casefold()}, or does not "
                "provide a reasonable basis to infer that it is limited to "
                "municipal governance, respond with 'No'. "
                "Start your response with 'Yes' or 'No' and explain briefly."
            ),
        )

        G.add_edge("is_city", "final", condition=llm_response_starts_with_no)
        G.add_edge(
            "is_city", "has_city_name", condition=llm_response_starts_with_yes
        )
        G.add_node(
            "has_city_name",
            prompt=(
                "Based on the legal text, is there clear and specific "
                "evidence that the ordinance applies specifically to "
                f"**{jurisdiction.full_name_the_prefixed}**? This could "
                "include a direct mention of "
                f"**{jurisdiction.subdivision_name}**, "
                "a title, heading, or citation indicating it's an ordinance "
                f"for {jurisdiction.full_subdivision_phrase_the_prefixed}, "
                "or other language that reasonably ties the text to "
                f"{jurisdiction.full_name_the_prefixed} specifically. "
                "Generic references such as 'the "
                f"{jurisdiction.type.casefold()}' or "
                f"'{jurisdiction.type} Zoning Administrator' are not "
                "sufficient on their own unless clearly linked to "
                f"{jurisdiction.full_name_the_prefixed}. "
                "Start your response with 'Yes' or 'No' and explain "
                "briefly."
            ),
        )
        G.add_edge("has_city_name", "final")

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
            f"{jurisdiction.type.casefold()}-wide) governed by "
            f"**{jurisdiction.full_name_the_prefixed}** "
            "(`false` otherwise). The value of the 'explanation' key should "
            "be a string containing a brief explanation for your choice. "
        ),
    )
    return G


def setup_graph_correct_jurisdiction_from_url(jurisdiction, **kwargs):
    """Setup graph to check for correct jurisdiction in URL

    Parameters
    ----------
    jurisdiction : Jurisdiction
        Jurisdiction for which validation is being performed.
    **kwargs
        Keyword-value pairs to add to graph.

    Returns
    -------
    networkx.DiGraph
        Graph instance that can be used to initialize an
        `elm.tree.DecisionTree`.
    """
    G = setup_graph_no_nodes(  # noqa: N806
        d_tree_name="Correct jurisdiction type from URL", **kwargs
    )

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
                f"{jurisdiction.full_subdivision_phrase_the_prefixed} in "
                "some way (e.g. either by full name or abbreviation)? **Do "
                "not** answer based on auxiliary information like state or "
                "county names. "
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


def _jurisdiction_names_to_extract(jurisdiction):
    """Determine whether jurisdiction name is required or not"""
    if not jurisdiction.subdivision_name and not jurisdiction.county:
        return "the state name"
    return f"the state name and the {jurisdiction.type.casefold()} name"
