.. include:: ../../../README.rst
   :start-after: inclusion-intro
   :end-before: Read more about the tool


Which local codes and ordinances does INFRA-COMPASS support?
============================================================
INFRA-COMPASS supports several extraction plugins (the value you set as ``tech`` in the
input config). Each plugin defines its own search queries, keyword heuristics, and field
set. Example fields extracted are listed below:

1. **Solar** (``tech: "solar"``) — utility-scale solar siting ordinances. Examples:
   setbacks (from structures, property lines, roads, railroads, transmission, water,
   conservation lands), maximum structure height, minimum/maximum lot size, maximum
   project size, panel spacing, land-use density, land coverage, noise, glare, visual
   impact, fencing, signage, screening, decommissioning, repowering, prohibitions, plus
   permitted-use district lists.

2. **Wind** (``tech: "wind"``) — utility-scale wind siting ordinances. Examples: setbacks
   (same feature set as solar), maximum turbine height, minimum/maximum lot size, maximum
   project size, separation from other wind energy conversion systems (WECS), shadow
   flicker, tower density, blade clearance, noise, color, lighting, decommissioning,
   repowering, climbing prevention, signage, visual impact, prohibitions.

3. **Small wind** (``tech: "small wind"``) — distributed / residential-scale wind ordinances.
   Examples: setbacks (extended to also include unoccupied structures), maximum turbine
   height, minimum/maximum lot size, rated capacity, blade clearance, tower density, shadow
   flicker, noise, color, lighting, decommissioning, climbing prevention, signage,
   prohibitions, permitting fees.

4. **Geothermal heat pumps** (``tech: "ghp"``) — GHP / ground-source heat-pump local codes.
   Examples: setbacks from driveways, property lines, yards, private/public water, building
   foundations, wastewater, water/sewer lines, animal enclosures, roads, rights-of-way (ROW),
   above/below-ground fuel, subsurface drains, wetlands, pools, hazardous materials;
   minimum/maximum well depth, noise, well/geothermal/GHP definitions, licensed-driller and
   certification requirements, screening, permits, inspections, decommissioning,
   prohibitions.

5. **Texas water rights** (``tech: "tx water rights"``) — multi-document extraction tailored
   to TX water-rights filings. Examples: permit requirements, extraction requirements, well
   spacing, drilling window, metering devices, district/well drought-management plans,
   plugging requirements, external-transfer requirements, production reporting, production
   cost, setbacks, redrilling, plus daily/monthly/annual withdrawal limits.

The following plugins are under active development:

6. **Geothermal electricity** (``tech: "geothermal"``) — Coming Soon.
7. **Transmission lines** (``tech: "transmission"``) — Coming Soon.
8. **Data centers** (``tech: "data centers"``) — Coming Soon.
9. **Natural gas** (``tech: "natural gas"``) — Coming Soon.

In addition, INFRA-COMPASS supports a **one-shot, schema-driven plugin** for arbitrary
ordinance types: provide a JSON schema for the fields you want and INFRA-COMPASS will use
LLM structured outputs to do a single-call extraction without writing decision trees. See
the `one-shot schema extraction example
<https://github.com/NatLabRockies/COMPASS/tree/main/examples/one_shot_schema_extraction>`_
for a walkthrough.


How does INFRA-COMPASS find the codes and ordinances?
=====================================================
A *jurisdiction* in INFRA-COMPASS is the place that issues the ordinance — typically a
U.S. county (or equivalent), e.g. "Boulder County, Colorado." Each run is driven by a CSV
of jurisdictions you want covered.

The retrieval flow is summarized below. INFRA-COMPASS first gathers candidate documents
from several possible sources, then uses an LLM-driven filter to keep only those that are
legally binding, pertain to the correct jurisdiction, and contain the relevant technology
regulations.

.. image:: https://raw.githubusercontent.com/NatLabRockies/COMPASS/main/docs/source/_static/document_retrieval.png
    :alt: Document retrieval flow. From a user-supplied jurisdiction (e.g. 'Jefferson County, Colorado'), Step 1 web-scrapes via search engine or website crawl; Step 2 downloads the document collection; Step 3 runs an LLM filter as a series of decision-tree questions ('Is the document a legal ordinance?', 'Does it pertain to the correct jurisdiction?', 'Does it contain relevant technology regulations?') to produce a confirmed ordinance document.
    :align: center
    :width: 100%

In practice, INFRA-COMPASS can find source documents in four different ways:

1. **Known local docs** — any documents you already have on hand for the jurisdiction.
2. **Known URLs** — links containing documents from which you want to pull data.
3. **Targeted document search** — plugin-defined queries that directly look for the
   **ordinance document itself**. Best for ordinances that are well-indexed by search
   engines as standalone PDFs or pages.
4. **Jurisdiction-website crawl** — for ordinances buried inside a county or municipal
   website. INFRA-COMPASS first finds the jurisdiction's official website, then crawls it
   for ordinance text using plugin-defined heuristics.


How does INFRA-COMPASS extract the data?
========================================
Once a candidate document is in hand, INFRA-COMPASS reduces it to just the passages that
actually pertain to the target technology, and then walks a small decision tree per
ordinance feature to read structured values out of those passages.

.. image:: https://raw.githubusercontent.com/NatLabRockies/COMPASS/main/docs/source/_static/document_extraction.png
    :alt: Document extraction flow. Step 1 (Ordinance Text Extraction) takes an ordinance document, splits it into text chunks, runs each chunk through an LLM content filter, and concatenates the relevant chunks into a single cleaned 'ordinance text.' Step 2 (Ordinance Value Extraction) feeds that cleaned text to a separate decision tree for each ordinance feature (e.g. Max Shadow Flicker, Property Line Setback, Road Setback, Structure Setback), producing structured ordinance data.
    :align: center
    :width: 100%


The full pipeline is designed to keep cost down and hallucinations out:

1. **Cheap keyword filter** — rejects sections that obviously aren't ordinance text *before*
   any LLM call.
2. **Legal-text validation** — an LLM classifies each surviving section as legally-binding
   regulation or not, with surrounding context for borderline cases.
3. **Relevant-text extraction** — each document is filtered down to just the passages that
   actually pertain to the target technology, producing a much shorter cleaned text that
   later steps work from.
4. **Hallucination guardrail** — the cleaned text is compared back to the original; if too
   much has been invented or paraphrased away, INFRA-COMPASS retries and ultimately drops
   the document rather than passing fabricated text to the next step.
5. **Decision-tree prompting** — rather than asking the LLM to fill out a whole row of
   values in one mega-prompt, INFRA-COMPASS extracts each value by walking a small decision
   tree of focused yes/no questions over the cleaned text. For a setback value, that might
   look like: "Does this text mention a setback from property lines for utility-scale
   solar?" → if yes, "Is the value given as a fixed distance, a multiple of system height,
   or relative to another feature?" → "What are the units?" → "What is the numeric value?"
   Each answer narrows the next question, and the conversation history carries forward so
   later prompts know what was already decided. This breaks an ambiguous extraction task
   into a sequence of unambiguous ones — which is what keeps structured output accurate
   even on messy ordinance language.
6. **Structured parsing** — extracted values are assembled into a tidy per-jurisdiction CSV
   row (one row of setbacks, height limits, lot sizes, and the rest), then merged across all
   jurisdictions into the final ordinance database — published as CSV, Excel workbook, and
   GeoPackage. Each row records the **source URL** of the document it came from, so any
   extracted value can be traced back to the original ordinance text and independently
   verified.

Jurisdictions are processed concurrently while respecting your API provider's rate limits,
with live cost tracking on a progress bar.


Obtaining ordinance data
========================
The National Laboratories of the Rockies (NLR) typically runs the INFRA-COMPASS pipeline
annually and publishes refreshed datasets to OpenEI.

If the technology or jurisdictions you need aren't in a published release — or you don't
want to wait for the next annual refresh — you can **run INFRA-COMPASS yourself**
on just the jurisdictions and technologies you care about.

**You bring your own LLM API key.** INFRA-COMPASS is built around OpenAI
(``client_type: "openai"``) and Azure OpenAI (``client_type: "azure"``). Set
``OPENAI_API_KEY`` (or the ``AZURE_OPENAI_*`` trio) in your environment before starting a
run. Support for other LLM providers is under consideration.


How can I expand INFRA-COMPASS to cover other ordinances?
=========================================================
Two paths, depending on how much customization you need:

- **Schema-driven (fastest)** — write a JSON schema describing the fields you want and pass
  it to ``compass process --plugin path/to/schema.json``. No Python required. Walkthrough:
  `one-shot schema extraction example
  <https://github.com/NatLabRockies/COMPASS/tree/main/examples/one_shot_schema_extraction>`_.
- **Full plugin (most control)** — implement a custom extraction plugin with your own search
  queries, heuristics, text collectors, and structured-data parsers. The
  `solar plugin <https://github.com/NatLabRockies/COMPASS/tree/main/compass/extraction/solar>`_
  is the cleanest reference to copy and adapt.

See the development guides for full details:
`plugin development <https://natlabrockies.github.io/COMPASS/dev/plugin_development.html>`_
and
`advanced plugin development <https://natlabrockies.github.io/COMPASS/dev/advanced_plugin_development.html>`_.

Validation
==========
Take a look at some high-level validation results in the `validation report <https://natlabrockies.github.io/COMPASS/val/validation.html>`_.
