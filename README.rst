*******************************************************************************************
Infrastructure Continuous Ordinance Mapping for Planning and Siting Systems (INFRA-COMPASS)
*******************************************************************************************

|License| |Zenodo| |PythonV| |PyPi| |Ruff| |Pixi| |SWR|

.. |PythonV| image:: https://badge.fury.io/py/INFRA-COMPASS.svg
    :target: https://pypi.org/project/INFRA-COMPASS/

.. |PyPi| image:: https://img.shields.io/pypi/pyversions/INFRA-COMPASS.svg
    :target: https://pypi.org/project/INFRA-COMPASS/

.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff

.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-orange.svg
    :target: https://opensource.org/licenses/BSD-3-Clause

.. |Pixi| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json
    :target: https://pixi.sh

.. |SWR| image:: https://img.shields.io/badge/SWR--25--62_-blue?label=NLR
    :alt: Static Badge

.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.17173409.svg
    :target: https://doi.org/10.5281/zenodo.17173409

.. inclusion-intro


INFRA-COMPASS is an innovative software tool that harnesses the power of Large Language Models (LLMs)
to automate the compilation and continued maintenance of an inventory of state and local codes
and ordinances pertaining to energy infrastructure.


Which local codes and ordinances does COMPASS support?
======================================================
COMPASS currently has five extraction plugins (the value you set as ``tech`` in the input
config). Each plugin defines its own search queries, keyword heuristics, and field set.
Example fields extracted are listed below:

1. **Solar** (``tech: "solar"``) — utility-scale solar siting ordinances. The most polished
   plugin and the one used by the quickstart demo. Examples: setbacks (from structures,
   property lines, roads, railroads, transmission, water, conservation lands), maximum
   structure height, minimum/maximum lot size, maximum project size, panel spacing, land-use
   density, land coverage, noise, glare, visual impact, fencing, signage, screening,
   decommissioning, repowering, prohibitions, plus permitted-use district lists.

2. **Wind** (``tech: "wind"``) — utility-scale wind siting ordinances. Examples: setbacks
   (same feature set as solar), maximum turbine height, minimum/maximum lot size, maximum
   project size, separation from other WECS, shadow flicker, tower density, blade clearance,
   noise, color, lighting, decommissioning, repowering, climbing prevention, signage, visual
   impact, prohibitions.

3. **Small wind** (``tech: "small wind"``) — distributed / residential-scale wind ordinances.
   Examples: setbacks (extended to also include unoccupied structures), maximum turbine
   height, minimum/maximum lot size, rated capacity, blade clearance, tower density, shadow
   flicker, noise, color, lighting, decommissioning, climbing prevention, signage,
   prohibitions, permitting fees.

4. **Geothermal heat pumps** (``tech: "ghp"``) — GHP / ground-source heat-pump local codes.
   Examples: setbacks from driveways, property lines, yards, private/public water, building
   foundations, wastewater, water/sewer lines, animal enclosures, roads, ROW,
   above/below-ground fuel, subsurface drains, wetlands, pools, hazardous materials;
   minimum/maximum well depth, noise, well/geothermal/GHP definitions, licensed-driller and
   certification requirements, screening, permits, inspections, decommissioning,
   prohibitions.

5. **Texas water rights** (``tech: "tx water rights"``) — multi-document extraction tailored
   to TX water-rights filings. Examples: permit requirements, extraction requirements, well
   spacing, drilling window, metering devices, district/well drought-management plans,
   plugging requirements, external-transfer requirements, production reporting, production
   cost, setbacks, redrilling, plus daily/monthly/annual withdrawal limits.

In addition, COMPASS supports a **one-shot, schema-driven plugin** for arbitrary ordinance
types: provide a JSON schema for the fields you want and COMPASS will use OpenAI structured
outputs to do a single-call extraction without writing decision trees. See
``examples/one_shot_schema_extraction/`` for a walkthrough.


How does COMPASS find the codes and ordinances?
================================================
A *jurisdiction* in COMPASS is the place that issues the ordinance — typically a U.S. county
(or equivalent) plus its state, e.g. "Boulder County, Colorado." Each run is driven by a CSV
of jurisdictions you want covered.

COMPASS finds source documents on the open web using **four-step retrieval** per jurisdiction:

1. **Known local docs** — any documents you pre-stage on disk for that jurisdiction.
2. **Known URLs** — direct links you've supplied in the jurisdiction config.
3. **Targeted document search** — plugin-defined queries that look for the **ordinance
   document itself** (e.g. ``"{jurisdiction} solar ordinance"``). The top hits are
   downloaded directly. Best for ordinances that are well-indexed by search engines as
   standalone PDFs or pages.
4. **Jurisdiction-website crawl** — for ordinances buried inside a county or municipal
   website. COMPASS first runs a *different* search (e.g. ``"{jurisdiction} website"``) to
   find the jurisdiction's official home page, asks an LLM to confirm the site really
   belongs to that jurisdiction, then does a BFS, keyword-prioritized crawl of that one
   site looking for ordinance pages.

**You bring your own LLM API key.** COMPASS is built around OpenAI (``client_type: "openai"``)
and Azure OpenAI (``client_type: "azure"``). Set ``OPENAI_API_KEY`` (or the
``AZURE_OPENAI_*`` trio) in your environment before starting a run. Anthropic support is
available as an optional extra (``pip install infra-compass[anthropic]``).


How does COMPASS extract the data?
==================================
Once a candidate document is downloaded, COMPASS runs a multi-step pipeline designed to keep
cost down and hallucinations out:

1. **Cheap keyword filter** — rejects sections that obviously aren't ordinance text *before*
   any LLM call.
2. **Legal-text validation** — an LLM classifies each surviving section as legally-binding
   regulation or not, with surrounding context for borderline cases.
3. **Decision-tree prompting** — rather than asking the LLM to fill out a whole row of values
   in one mega-prompt, COMPASS extracts each value by walking a small decision tree of
   focused yes/no questions. For a setback value, that might look like: "Does this text
   mention a setback from property lines for utility-scale solar?" → if yes, "Is the value
   given as a fixed distance, a multiple of system height, or relative to another feature?"
   → "What are the units?" → "What is the numeric value?" Each answer narrows the next
   question, and the conversation history carries forward so later prompts know what was
   already decided. This breaks an ambiguous extraction task into a sequence of unambiguous
   ones — which is what keeps structured output accurate even on messy ordinance language.
4. **Hallucination guardrail** — cleaned text is compared back to the original; if too much
   has been invented or paraphrased away, COMPASS retries and ultimately drops the document
   rather than recording a fabricated value.
5. **Structured parsing** — extracted values are assembled into a tidy per-jurisdiction CSV
   row (one row of setbacks, height limits, lot sizes, and the rest), then merged across all
   jurisdictions into a single output database at the end of the run. Each row records the
   **source URL** of the document it came from, so any extracted value can be traced back to
   the original ordinance text and independently verified.

Jurisdictions are processed concurrently while respecting your API provider's rate limits,
with live cost tracking on a progress bar.


Where is the data stored and how is it maintained?
===================================================
The latest published ordinance datasets for solar and wind are available here:

- Solar: https://data.openei.org/submissions/8519
- Wind: https://data.openei.org/submissions/8602

NLR typically runs the COMPASS pipeline annually and publishes refreshed datasets to OpenEI.


How can I expand COMPASS to cover other ordinances?
====================================================
Two paths, depending on how much customization you need:

- **Schema-driven (fastest)** — write a JSON schema describing the fields you want and pass
  it to ``compass process --plugin <schema.json>``. No Python required. Walkthrough:
  ``examples/one_shot_schema_extraction/``.
- **Full plugin (most control)** — implement a custom extraction plugin with your own search
  queries, heuristics, text collectors, and structured-data parsers. The ``solar`` plugin is
  the cleanest reference to copy and adapt.

See the development guides for details:
`plugin development <https://natlabrockies.github.io/COMPASS/dev/plugin_development.html>`_
and
`advanced plugin development <https://natlabrockies.github.io/COMPASS/dev/advanced_plugin_development.html>`_.


Installing INFRA-COMPASS
========================
The quickest way to install INFRA-COMPASS for users is from PyPi:

.. code-block:: bash

    pip install infra-compass

If you would like to install and run INFRA-COMPASS from source, we recommend using `pixi <https://pixi.sh/latest/>`_:

.. code-block:: bash

    git clone git@github.com:NatLabRockies/COMPASS.git; cd COMPASS
    pixi run compass

Before performing any web searches (i.e. running the COMPASS pipeline), you will need to run the following command:

.. code-block:: shell

    playwright install

If you are using ``pixi``, don't forget to prefix this command with ``pixi run`` or initialize a shell using ``pixi shell``.
For detailed instructions, see the `installation documentation <https://natlabrockies.github.io/COMPASS/misc/installation.html>`_.


Quickstart
==========
To run a quick INFRA-COMPASS demo, set up a personal OpenAI API key and run:

.. code-block:: shell

    pixi run openai-solar-demo <your API key>

This will run a full extraction pipeline for two counties using ``gpt-4o-mini`` (costs ~$0.45).
For more information on configuring an INFRA-COMPASS run, see the
`execution basics example <https://natlabrockies.github.io/COMPASS/examples/execution_basics/README.html>`_.


Development
===========
Please see the `Development Guidelines <https://natlabrockies.github.io/COMPASS/dev/index.html>`_
if you wish to contribute code to this repository.
