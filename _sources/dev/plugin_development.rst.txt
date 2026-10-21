.. _plugin_development_tutorial:

Ordinance Plugin Development Tutorial
=====================================

This tutorial walks you through building a battery storage extraction
plugin. The goal is not just to show what to type, but to explain how and
why each piece of a COMPASS plugin works. By the end, you will have a
complete, registered plugin and a mental model of the data flow that makes
extraction reliable.

What you will build
-------------------

A complete ``battery_storage`` plugin that can be discovered by the CLI,
can filter documents quickly, and can turn ordinance text into structured
outputs. Along the way you will see how COMPASS uses a label chain to pass
data from one stage to the next.

You will build:

- A keyword heuristic for fast filtering
- An LLM text collector and extractor
- Decision-tree parsers for structured outputs
- A registered plugin that runs with ``compass process``


Choose the base class
---------------------

Start by choosing the base class that matches how much control you need.
For ordinance technologies that look like solar or wind, start with
``OrdinanceExtractionPlugin``. It wires together filtering, collection,
extraction, and CSV output so you can focus on domain logic.

Reach for ``FilteredExtractionPlugin`` when filtering can remain standard,
but you need custom parsing logic. Choose ``BaseExtractionPlugin`` only when
you want to control the whole workflow yourself, including filtering and
output handling. Think of this choice as your contract with the framework:
the more you want to customize, the more responsibilities you accept.

Project skeleton
----------------

Create a new extraction package. This keeps the pieces close to each other
so you can reason about the full pipeline as you build it.

.. code-block:: text

   compass/extraction/battery_storage/
       __init__.py
       plugin.py
       ordinance.py
       parse.py
       graphs.py

How the pieces fit
------------------

The pipeline is a conversation between components. Each one writes its
results to a label, and the next component reads from that label. If the
labels are misaligned, the data disappears and the parser will never see
the text you extracted.

.. code-block:: text

    Document.text -> collector.OUT_LABEL -> extractor.OUT_LABEL -> parser.IN_LABEL

This label chain is the most important mental model to keep while you are
building and debugging. You will revisit it in the troubleshooting guide.

Step 1: Heuristic
-----------------

In ``ordinance.py``, define a keyword heuristic to quickly screen out
non-relevant documents before LLM calls. Think of this as the bouncer at
the door: it keeps obvious non-target content out so you spend model budget
only where it matters.

.. code-block:: python

   from compass.plugin import KeywordBasedHeuristic

   class BatteryStorageHeuristic(KeywordBasedHeuristic):

       NOT_TECH_WORDS = [
           "car battery",
           "vehicle battery",
           "phone battery",
           "laptop battery",
           "battery lawsuit",
       ]

       GOOD_TECH_KEYWORDS = [
           "bess",
           "battery",
           "storage",
           "lithium",
           "capacity",
       ]

       GOOD_TECH_ACRONYMS = ["BESS", "ESS"]

       GOOD_TECH_PHRASES = [
           "battery energy storage",
           "battery storage system",
           "energy storage system",
       ]

Step 2: Text collector
----------------------

In ``ordinance.py``, define a collector that keeps only chunks that
describe actual regulations. The collector answers a simple question:
should this chunk move forward in the pipeline, or be discarded? You want
to be conservative here, keeping the chunks that contain real regulatory
requirements.

.. code-block:: python

   from compass.plugin import PromptBasedTextCollector

   class BatteryStorageTextCollector(PromptBasedTextCollector):

       IN_LABEL = "text"
       OUT_LABEL = "battery_storage_relevant_text"

       PROMPTS = [
           {
               "prompt": (
                   "Does this text discuss regulations or zoning rules for "
                   "battery energy storage systems (BESS)? Return JSON with "
                   "key '{key}' set to true/false. Ignore consumer batteries."
               ),
               "key": "discusses_battery_storage_regulations",
               "label": "battery storage regulation check",
           },
           {
               "prompt": (
                   "Does this text contain specific requirements like "
                   "setbacks, capacity limits, safety standards, or "
                   "permit procedures? Return JSON with key '{key}' set to "
                   "true/false"
               ),
               "key": "contains_specific_requirements",
               "label": "specific requirements check",
           },
       ]

Step 3: Text extractor
----------------------

In ``ordinance.py``, define an extractor that condenses the retained
chunks into a clean extraction text. This step is where you reduce a messy
set of regulatory paragraphs into a focused summary that a parser can
reason over. The output label becomes the parser input, so clarity here
directly improves downstream accuracy.

.. code-block:: python

   from compass.plugin import PromptBasedTextExtractor

   class BatteryStorageTextExtractor(PromptBasedTextExtractor):

       IN_LABEL = "battery_storage_relevant_text"

       PROMPTS = [
           {
               "prompt": (
                   "Extract battery storage regulations. Include setbacks, "
                   "capacity limits, height limits, safety requirements, "
                   "fire suppression, screening, noise, and permits. "
                   "{FORMATTING_PROMPT} {OUTPUT_PROMPT}"
               ),
               "key": "battery_storage_ordinance_text",
               "out_fn": "{jurisdiction}_battery_summary.txt",
           }
       ]

Step 4: Parsers and decision trees
----------------------------------

In ``parse.py``, define parsers that run decision trees against the
extraction text. The trees live in ``graphs.py``. This is the moment where
your plugin turns narrative text into structured fields, so it helps to
think in terms of a guided interview: the tree asks focused questions and
routes to the right follow-up based on answers.

.. code-block:: python

   from compass.plugin import OrdinanceParser
   from compass.common import setup_async_decision_tree, run_async_tree
   from compass.extraction.battery_storage.graphs import (
       setup_graph_battery_setbacks,
       setup_graph_capacity_limits,
   )
   import pandas as pd

   class BatteryStorageParser(OrdinanceParser):

       IN_LABEL = "battery_storage_ordinance_text"
       OUT_LABEL = "battery_storage_ordinance_values"

       async def parse(self, text):
           chat_caller = self._init_chat_llm_caller(
               "You extract battery storage regulations from ordinance text"
           )

           results = {}

           setback_tree = setup_async_decision_tree(
               setup_graph_battery_setbacks,
               text=text,
               chat_llm_caller=chat_caller,
           )
           results["setbacks"] = await run_async_tree(setback_tree)

           capacity_tree = setup_async_decision_tree(
               setup_graph_capacity_limits,
               text=text,
               chat_llm_caller=chat_caller,
           )
           results["capacity"] = await run_async_tree(capacity_tree)

           if not any(results.values()):
               return None

           return pd.DataFrame([results])

Step 5: Wire the plugin
-----------------------

In ``plugin.py``, connect the components and register the plugin. The
registration step is what makes your class discoverable by the CLI, so do
not skip it. You are effectively telling COMPASS, "this identifier maps to
this pipeline."

.. code-block:: python

   from compass.plugin import OrdinanceExtractionPlugin, register_plugin
   from compass.extraction.battery_storage.ordinance import (
       BatteryStorageHeuristic,
       BatteryStorageTextCollector,
       BatteryStorageTextExtractor,
   )
   from compass.extraction.battery_storage.parse import BatteryStorageParser

   BatteryStorageParser.IN_LABEL = BatteryStorageTextExtractor.OUT_LABEL

   class BatteryStorageExtractor(OrdinanceExtractionPlugin):

       IDENTIFIER = "battery_storage"

       QUESTION_TEMPLATES = [
           "battery energy storage ordinance {jurisdiction}",
           "{jurisdiction} battery storage regulations",
           "{jurisdiction} BESS zoning",
           "site:{jurisdiction_website} battery storage ordinance",
       ]

       WEBSITE_KEYWORDS = {
           "ordinance": 1000,
           "battery": 900,
           "storage": 800,
           "bess": 900,
           "energy storage": 850,
       }

       heuristic = BatteryStorageHeuristic()

       TEXT_COLLECTORS = [BatteryStorageTextCollector]
       TEXT_EXTRACTORS = [BatteryStorageTextExtractor]
       PARSERS = [BatteryStorageParser]

   register_plugin(BatteryStorageExtractor)

Query templates must use ``{jurisdiction}`` where the jurisdiction's full
name should appear. They may also use the exact placeholder
``{jurisdiction_website}`` to restrict a query to the jurisdiction's known
website, as shown by the ``site:`` query above. COMPASS submits templates
containing this website placeholder only when the jurisdiction has a known
website URL; otherwise, it discards those templates before searching.

Step 6: Run it
--------------

At this point you have a complete pipeline: a heuristic to filter, a
collector to keep relevant chunks, an extractor to shape them, and a parser
to produce structured outputs. Running the CLI now exercises the full
story you just built.

.. code-block:: bash

   export OPENAI_API_KEY="your-key"

   compass process --config my_config.yaml

Where to go next
----------------

- Run the pipeline against a second jurisdiction to see how your heuristic and parser choices generalize
- Review the troubleshooting guidance in this tutorial to debug label chain issues quickly
- Explore the solar and wind plugins in the repository for more real-world patterns
- Step into :ref:`advanced_plugin_development` when you need custom filtering, RAG-style parsing, or nonstandard output schemas

Each step builds on the same flow you used here—filter, collect, extract,
parse, save. Move to the next level only when the current pattern no
longer fits your domain.


