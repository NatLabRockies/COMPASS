.. _advanced_plugin_development:

Advanced Plugin Development
===========================

The basic tutorial showed you the standard path: keyword filtering finds your
documents, collectors narrow them down, extractors shape the text, parsers
produce values.

But what happens when keyword filtering fails? When a single regulation
references three different PDFs? When you need to ask "show me all power
capacity requirements" across 200 pages of technical appendices? When
your output schema needs jurisdiction-level aggregation that standard
CSV writes cannot handle?

This guide covers how to handle those cases while keeping the same
overall flow.

When to go custom
-----------------

Standard patterns work until they no longer meet the requirements of your
domain. The signals below indicate when you need more control:

**Multi-document complexity**
  Regulations split across zoning codes, technical standards, and special
  use permits. Keyword filtering cannot see the connections. You need
  semantic search across the full corpus.

**Domain-specific preprocessing**
  Your PDFs embed critical metadata in filenames or document properties.
  Standard collectors ignore this. You need to extract and normalize
  before filtering runs.

**Parallel extraction requirements**
  You are extracting 15+ features and each LLM call takes 3 seconds.
  Sequential parsing means 45+ seconds per document. You need async
  orchestration.

**Custom output schemas**
  Your schema has nested relationships, jurisdiction-level rollups, or
  domain-specific validation. The standard DataFrame→CSV pattern cannot
  express your data model.

This guide will take you from the standard ``OrdinanceExtractionPlugin``
through progressively more custom implementations until you control the
entire extraction pipeline.

The scenario: data center ordinances
-------------------------------------

We will build a plugin for data center facility regulations. These
ordinances are perfect for demonstrating advanced patterns because they
exhibit all the complexity signals:

- Regulations span zoning codes (setbacks), electrical codes (power
  capacity), building codes (cooling systems), and environmental codes
  (noise limits)
- Power capacity values often appear in PDF metadata or appendix titles
- Feature extraction benefits from parallel queries: power requirements,
  cooling methods, generator rules, screening requirements, noise limits,
  height restrictions, etc.
- Output needs facility type classification and cross-jurisdiction
  comparison of infrastructure requirements

By the end you will have a complete custom plugin and a mental model for
when each customization level makes sense.

Level 1: Extending with hooks
------------------------------

Before you redesign the pipeline, try the extension points first.
``FilteredExtractionPlugin`` gives you hooks—moments where you can insert
custom logic while keeping the machinery running. Hooks serve as targeted
annotations within standard filtering.

The two hooks available:

``pre_filter_docs_hook(extraction_context)``
  Runs before filtering begins. Use this to preprocess documents, extract
  metadata, normalize jurisdiction names, or enrich the document objects
  with custom attributes.

``post_filter_docs_hook(extraction_context)``
  Runs after filtering completes but before parsing. Use this to validate
  filtering results, add cross-document references, or prepare extraction
  state.

Example: extracting power capacity from PDF metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data center ordinances often include power capacity thresholds in
document titles: ``"Zoning Amendment for Facilities Over 50MW.pdf"``.
This metadata should influence filtering, but standard collectors never
see it.

.. code-block:: python

   from compass.plugin import FilteredExtractionPlugin
   from compass.extraction.data_center.ordinance import (
       DataCenterHeuristic,
       DataCenterTextCollector,
       DataCenterTextExtractor,
   )
   import re

   class DataCenterExtractorWithHooks(FilteredExtractionPlugin):

       IDENTIFIER = "data_center_hooks"

       heuristic = DataCenterHeuristic()
       TEXT_COLLECTORS = [DataCenterTextCollector]
       TEXT_EXTRACTORS = [DataCenterTextExtractor]

       async def pre_filter_docs_hook(self, extraction_context):
           for doc in extraction_context.docs:
               capacity_match = re.search(
                   r"(\d+)\s*(MW|megawatts?)",
                   doc.metadata.get("source", ""),
                   re.IGNORECASE,
               )
               if capacity_match:
                   capacity_mw = float(capacity_match.group(1))
                   doc.metadata["extracted_capacity_mw"] = capacity_mw

               if "data center" in doc.metadata.get("source", "").lower():
                   doc.metadata["explicitly_about_data_centers"] = True

       def parse_docs_for_structured_data(self, doc_infos, out_dir):
           return None

The hook runs before any LLM calls. It scans PDF filenames, extracts
numeric capacity values, and marks documents that explicitly mention data
centers. Later stages can read these attributes from ``doc.metadata`` and
make informed decisions.

When hooks suffice
^^^^^^^^^^^^^^^^^^

Hooks work well when you need to:

- Normalize jurisdiction name variations before filtering
- Extract document dates or revision numbers from metadata
- Flag high-priority documents based on filename patterns
- Enrich documents with external data (GIS lookups, jurisdiction type)
- Log or validate document counts at pipeline stages

Hooks preserve the standard filtering pipeline while giving you control
at key moments. They do not change how filtering itself operates. When
you need to alter the filtering mechanics, move to a deeper level of
customization.

Level 2: Custom filtering with BaseExtractionPlugin
----------------------------------------------------

Standard filtering is a progressive narrowing: keywords reduce the
universe, collectors prune chunks, extractors shape what remains. This
works when you can define your target domain with explicit patterns.

But semantic relationships do not obey keyword rules. When you need to
answer questions like "what are the generator requirements?" or "how is
cooling system noise regulated?", you need retrieval.

In these cases, use ``BaseExtractionPlugin`` to gain full control of the
filtering pipeline. It requires you to implement the entire pipeline
yourself. There are no collectors, extractors, or automatic label chain
wiring. You start with raw documents and produce structured data while
controlling each step.

The contract
^^^^^^^^^^^^

``BaseExtractionPlugin`` requires you to implement:

``get_query_templates()``
  Return list of search query templates for document discovery.

``get_website_keywords()``
  Return dict of keywords→priority for web crawling.

``get_heuristic()``
  Return a heuristic instance for initial document screening.

``filter_docs(extraction_context)``
  The heart of customization. Take ``extraction_context.docs`` and reduce
  them to relevant content. Store anything you want in
  ``extraction_context.attrs`` for later stages.

``parse_docs_for_structured_data(doc_infos, out_dir)``
  Extract structured values and return results suitable for saving.

``save_structured_data(doc_infos, out_dir)``
  Write results to disk in whatever format makes sense for your domain.

This is a contract between you and the framework. Implement these six
methods and COMPASS will run your plugin. What you do inside them is
entirely up to you.

Building a document corpus with embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The key insight for complex extraction is to stop progressive filtering
and start building corpora. Instead of narrowing 100 documents to 5,
embed all 100 and let semantic search find relevant chunks at extraction
time.

This pattern comes from the water rights plugin. The following adapts it
for data centers:

.. code-block:: python

    import asyncio
    import os
    import numpy as np
    import pandas as pd
    from openai import AzureOpenAI
    from compass.plugin import BaseExtractionPlugin, register_plugin
    from compass.plugin.heuristic import NoOpHeuristic

    class DataCenterExtractorCustom(BaseExtractionPlugin):

        IDENTIFIER = "data_center_custom"

        QUESTION_TEMPLATES = [
            "data center ordinance {jurisdiction}",
            "{jurisdiction} data center zoning regulations",
            "{jurisdiction} server farm power requirements",
            "{jurisdiction} cooling system noise limits",
        ]

        WEBSITE_KEYWORDS = {
            "ordinance": 1000,
            "data center": 950,
            "server": 900,
            "colocation": 900,
            "facility": 800,
            "power": 850,
            "cooling": 850,
            "generator": 800,
        }

        @classmethod
        def get_query_templates(cls):
            return cls.QUESTION_TEMPLATES

        @classmethod
        def get_website_keywords(cls):
            return cls.WEBSITE_KEYWORDS

        @classmethod
        def get_heuristic(cls):
            return NoOpHeuristic()

        async def filter_docs(self, extraction_context):
            docs = extraction_context.docs

            page_texts = [
                page.text
                for doc in docs
                for page in doc.pages
                if page.text.strip()
            ]

            if not page_texts:
                extraction_context.attrs["corpus"] = pd.DataFrame(
                    columns=["text", "embedding"]
                )
                return

            client = AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version="2024-10-21",
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            )

            rate_limit = asyncio.Semaphore(5)

            async def embed_with_limit(text):
                async with rate_limit:
                    return await asyncio.to_thread(
                        client.embeddings.create,
                        model="text-embedding-3-large-standard",
                        input=text,
                    )

            embedding_tasks = [
                embed_with_limit(text) for text in page_texts
            ]
            embeddings = await asyncio.gather(*embedding_tasks)

            corpus = pd.DataFrame({
                "text": page_texts,
                "embedding": [
                    np.array(resp.data[0].embedding, dtype=np.float32)
                    for resp in embeddings
                ],
            })

            extraction_context.attrs["corpus"] = corpus

The filter method now runs an embedding pipeline that converts every page
into a vector. The resulting corpus lives in
``extraction_context.attrs["corpus"]``, and downstream stages query it
semantically.

The ``NoOpHeuristic`` keeps the pipeline permissive when you embed every
page. Keyword filtering offers no benefit in this mode, so the heuristic
allows all documents to reach the embedding stage.

Rate limiting remains essential because Azure OpenAI embedding endpoints
enforce request limits. The semaphore prevents overwhelming the service.
Adjust the concurrency limit to match your quota and the size of the
corpus.

When custom filtering makes sense
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build your own filter pipeline when:

- Semantic search would outperform keyword patterns
- You need to build cross-document indexes or reference graphs
- Document structure requires custom parsing before chunking
- You want to cache embeddings across runs
- Standard collectors are too slow or too restrictive

Filtering alone does not complete the extraction task. You still need
parsers that can query the corpus effectively; that is the next level of
customization.

Level 3: RAG-based extraction
------------------------------

You have a corpus of embedded chunks. Now you need parsers that can ask
questions of it. This is retrieval-augmented generation: query the vector
database, get relevant chunks, pass them to a decision tree.

This shift is significant. Standard parsers receive extraction text—a
single string containing "relevant" content—whereas RAG parsers receive a
query interface that returns relevant chunks on demand.

This means your parser can ask multiple questions, refine queries based on
initial results, and explore the corpus dynamically. It moves from
reading a summary to interrogating the documents directly.

Setting up a simple vector index
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can build a lightweight in-memory index with standard tools: OpenAI
embeddings for vectors and NumPy for similarity search. No additional
frameworks are required.

.. code-block:: python

   import os
   import numpy as np
   from openai import AzureOpenAI

   def _build_index(corpus):
       client = AzureOpenAI(
           api_key=os.environ["AZURE_OPENAI_API_KEY"],
           api_version="2024-10-21",
           azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
       )

       texts = corpus["text"].tolist()
       vectors = []

       for text in texts:
           resp = client.embeddings.create(
               model="text-embedding-3-large-standard",
               input=text,
           )
           vec = np.array(resp.data[0].embedding, dtype=np.float32)
           vec /= np.linalg.norm(vec) + 1e-9
           vectors.append(vec)

       return {
           "texts": texts,
           "vectors": np.vstack(vectors),
       }

   def _query_index(index, query, top_k=5):
       client = AzureOpenAI(
           api_key=os.environ["AZURE_OPENAI_API_KEY"],
           api_version="2024-10-21",
           azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
       )

       resp = client.embeddings.create(
           model="text-embedding-3-large-standard",
           input=query,
       )
       q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
       q_vec /= np.linalg.norm(q_vec) + 1e-9

       scores = index["vectors"] @ q_vec
       top_idx = scores.argsort()[::-1][:top_k]

       return [index["texts"][i] for i in top_idx]

Each call to ``_query_index`` performs a semantic search and returns the
top matching chunks.

Generic RAG extraction helper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Standard parsers run decision trees against static text. RAG parsers run
decision trees against query results. The pattern is to wrap tree
execution in a helper that fetches context first:

.. code-block:: python

   from compass.common import setup_async_decision_tree, run_async_tree

   async def _extract_with_rag(
       index,
       query,
       tree_setup_fn,
       chat_llm_caller,
   ):
       results = _query_index(index, query, top_k=5)

       if not results:
           return None

       context_text = "\n\n".join(results)

       tree = setup_async_decision_tree(
           tree_setup_fn,
           text=context_text,
           chat_llm_caller=chat_llm_caller,
       )

       return await run_async_tree(tree)

This helper encapsulates the RAG pattern: query → retrieve → extract. Call
it once per feature; it handles the mechanics of search and tree execution.

Parallel async parser
^^^^^^^^^^^^^^^^^^^^^

With the helper in place, your parser becomes a coordinator. For each
feature you want to extract, spawn a RAG query. Run them all in parallel
with ``asyncio.gather()``.

.. code-block:: python

   from compass.extraction.data_center.graphs import (
       setup_power_requirements_graph,
       setup_cooling_system_graph,
       setup_noise_limits_graph,
       setup_generator_rules_graph,
       setup_setback_requirements_graph,
       setup_screening_requirements_graph,
       setup_height_limits_graph,
   )

   class DataCenterParser:

       async def parse(self, extraction_context):
           corpus = extraction_context.attrs.get("corpus")
           if corpus is None or corpus.empty:
               return None

           index = _build_index(corpus)

           system_msg = (
               "You extract data center facility regulations from "
               "ordinance text. Focus on power, cooling, noise, and "
               "setback requirements."
           )

           chat_llm_caller = self._init_chat_llm_caller(system_msg)

           tasks = [
               _extract_with_rag(
                   index,
                   "What are the power capacity requirements?",
                   setup_power_requirements_graph,
                   chat_llm_caller,
               ),
               _extract_with_rag(
                   index,
                   "What cooling system regulations apply?",
                   setup_cooling_system_graph,
                   chat_llm_caller,
               ),
               _extract_with_rag(
                   index,
                   "What are the noise limits for data centers?",
                   setup_noise_limits_graph,
                   chat_llm_caller,
               ),
               _extract_with_rag(
                   index,
                   "What generator rules apply to data centers?",
                   setup_generator_rules_graph,
                   chat_llm_caller,
               ),
               _extract_with_rag(
                   index,
                   "What setback distances are required?",
                   setup_setback_requirements_graph,
                   chat_llm_caller,
               ),
               _extract_with_rag(
                   index,
                   "What screening or landscaping is required?",
                   setup_screening_requirements_graph,
                   chat_llm_caller,
               ),
               _extract_with_rag(
                   index,
                   "What height restrictions apply?",
                   setup_height_limits_graph,
                   chat_llm_caller,
               ),
           ]

           results = await asyncio.gather(*tasks)

           feature_names = [
               "power_capacity",
               "cooling_system",
               "noise_limit",
               "generator_rules",
               "setback",
               "screening",
               "height_limit",
           ]

           extracted = {
               name: result
               for name, result in zip(feature_names, results)
               if result is not None
           }

           if not extracted:
               return None

           return pd.DataFrame([extracted])

This parser issues seven feature-specific queries in a single
asynchronous batch. If each query takes 3 seconds, sequential execution
would take roughly 21 seconds. Running them in parallel completes in
about 3 seconds, and the speedup scales with the number of features.

The tradeoff is complexity. You need to manage async tasks, handle partial
failures gracefully, and ensure the RAG is thread-safe. For domains
with many features, the performance gain often justifies the effort.

Implementing RAG in parse_docs_for_structured_data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wire the parser into your plugin's extraction method:

.. code-block:: python

   class DataCenterExtractorCustom(BaseExtractionPlugin):

       async def parse_docs_for_structured_data(
           self, doc_infos, out_dir
       ):
           parser = DataCenterParser()

           extraction_context = doc_infos[0].extraction_context
           result_df = await parser.parse(extraction_context)

           if result_df is None or result_df.empty:
               return None

           jurisdiction = extraction_context.attrs.get("jurisdiction")
           result_df["jurisdiction"] = jurisdiction

           return result_df

Because ``BaseExtractionPlugin`` requires you to implement everything, you
control when and how the parser runs. Standard plugins call parsers
automatically; custom plugins call them explicitly.

When RAG makes sense
^^^^^^^^^^^^^^^^^^^^

Use RAG-based extraction when:

- Feature extraction requires answering specific questions
- Relevant context is scattered across many pages
- You need to query the same corpus multiple times with different prompts
- Standard extraction text is too broad or too narrow
- Performance demands parallel query execution

RAG trades setup complexity for query flexibility. You pay the embedding
cost once, then search is fast and targeted.

Level 4: Custom output and aggregation
---------------------------------------

Standard COMPASS plugins output CSV with columns like ``feature``,
``value``, ``units``, ``section``. This schema works for most ordinance
extractions, but not for all domains.

Data center regulations need a different schema. Facility types matter
(colocation vs enterprise vs edge). Power capacity is a primary key, not
just another feature. Cross-jurisdiction comparison requires aggregation
logic that standard saves cannot express.

When you implement ``save_structured_data()`` yourself, you control the
output format completely.

Custom schema design
^^^^^^^^^^^^^^^^^^^^

Start by defining what your output should represent. For data centers:

- **Facility type**: Colocation, enterprise, hyperscale, edge
- **Power capacity**: Minimum threshold in MW
- **Cooling method**: Air, liquid, hybrid, unspecified
- **Grid requirements**: Substation, redundant feeds, backup generation
- **Setback distances**: Property line, residential, right-of-way
- **Noise limits**: Day, night, measurement method
- **Height limit**: Maximum structure height
- **Screening requirements**: Fencing, landscaping, visual barriers

Each row is a jurisdiction, each column is a feature. This is a different
data model than the standard ``feature, value`` schema.

Implementing save_structured_data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Override the method to build your schema and write your format:

.. code-block:: python

   from pathlib import Path

   class DataCenterExtractorCustom(BaseExtractionPlugin):

       @classmethod
       def save_structured_data(cls, doc_infos, out_dir):
           results = []

           for doc_info in doc_infos:
               if doc_info.structured_data is None:
                   continue

               df = doc_info.structured_data

               jurisdiction = df.get("jurisdiction", [None])[0]

               row = {
                   "jurisdiction": jurisdiction,
                   "state": doc_info.extraction_context.attrs.get(
                       "state"
                   ),
                   "facility_type": df.get("facility_type", [None])[0],
                   "power_capacity_mw": df.get(
                       "power_capacity", [None]
                   )[0],
                   "cooling_method": df.get(
                       "cooling_system", [None]
                   )[0],
                   "grid_requirements": df.get(
                       "grid_connection", [None]
                   )[0],
                   "setback_property_line_ft": df.get(
                       "setback", [None]
                   )[0],
                   "noise_limit_day_db": df.get(
                       "noise_limit", [None]
                   )[0],
                   "height_limit_ft": df.get("height_limit", [None])[0],
                   "screening_required": df.get("screening", [None])[0],
               }

               results.append(row)

           if not results:
               return

           output_df = pd.DataFrame(results)

           output_path = Path(out_dir) / "data_center_ordinances.csv"
           output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

This method aggregates across jurisdictions, normalizes column names,
handles missing values, and writes a custom filename. You control the
entire output pipeline.

Alternative output formats
^^^^^^^^^^^^^^^^^^^^^^^^^^

CSV is not the only option. You can write JSON for nested structures,
Parquet for large datasets, or even push directly to a database:

.. code-block:: python

   import json

   @classmethod
   def save_structured_data(cls, doc_infos, out_dir):
       results = {}

       for doc_info in doc_infos:
           if doc_info.structured_data is None:
               continue

           jurisdiction = doc_info.extraction_context.attrs.get(
               "jurisdiction"
           )

           results[jurisdiction] = {
               "facility_requirements": {
                   "power": doc_info.structured_data.get(
                       "power_capacity"
                   ),
                   "cooling": doc_info.structured_data.get(
                       "cooling_system"
                   ),
               },
               "spatial_requirements": {
                   "setbacks": doc_info.structured_data.get("setback"),
                   "screening": doc_info.structured_data.get("screening"),
                   "height": doc_info.structured_data.get("height_limit"),
               },
               "operational_limits": {
                   "noise": doc_info.structured_data.get("noise_limit"),
                   "generator": doc_info.structured_data.get(
                       "generator_rules"
                   ),
               },
           }

       output_path = Path(out_dir) / "data_center_ordinances.json"
       with open(output_path, "w") as f:
           json.dump(results, f, indent=2)

JSON lets you express hierarchical relationships that CSV flattens.
Choose the format that matches your downstream use case.

When custom output makes sense
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implement your own save logic when:

- Your schema has nested or relational structure
- You need cross-jurisdiction aggregation or rollups
- Output format is not CSV (JSON, Parquet, database)
- Downstream tools expect specific column names or data types
- You want to validate or enrich results before saving

Custom output is the final layer of control. It ensures your extraction
pipeline produces the data structure your domain requires.

Complete plugin architecture
-----------------------------

With the individual components defined, you can now assemble them into a
complete plugin.

The full implementation requires coordinating heuristics, filtering,
parsing, and output. Here is the complete plugin class:

.. code-block:: python

    import asyncio
    import os
    import numpy as np
    import pandas as pd
    from openai import AzureOpenAI
    from pathlib import Path
    from compass.plugin import BaseExtractionPlugin, register_plugin
    from compass.plugin.heuristic import NoOpHeuristic

   class DataCenterExtractorCustom(BaseExtractionPlugin):

       IDENTIFIER = "data_center_custom"

       QUESTION_TEMPLATES = [
           "data center ordinance {jurisdiction}",
           "{jurisdiction} data center facility regulations",
           "{jurisdiction} server farm zoning requirements",
       ]

       WEBSITE_KEYWORDS = {
           "ordinance": 1000,
           "data center": 950,
           "facility": 800,
           "power": 850,
           "cooling": 850,
       }

       @classmethod
       def get_query_templates(cls):
           return cls.QUESTION_TEMPLATES

       @classmethod
       def get_website_keywords(cls):
           return cls.WEBSITE_KEYWORDS

       @classmethod
       def get_heuristic(cls):
           return NoOpHeuristic()

       async def filter_docs(self, extraction_context):
           docs = extraction_context.docs

           page_texts = [
               page.text
               for doc in docs
               for page in doc.pages
               if page.text.strip()
           ]

           if not page_texts:
               extraction_context.attrs["corpus"] = pd.DataFrame(
                   columns=["text", "embedding"]
               )
               return

           client = AzureOpenAI(
               api_key=os.environ["AZURE_OPENAI_API_KEY"],
               api_version="2024-10-21",
               azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
           )

           rate_limit = asyncio.Semaphore(5)

           async def embed_with_limit(text):
               async with rate_limit:
                   return await asyncio.to_thread(
                       client.embeddings.create,
                       model="text-embedding-3-large-standard",
                       input=text,
                   )

           embedding_tasks = [
               embed_with_limit(text) for text in page_texts
           ]
           embeddings = await asyncio.gather(*embedding_tasks)

           corpus = pd.DataFrame({
               "text": page_texts,
               "embedding": [
                   np.array(resp.data[0].embedding, dtype=np.float32)
                   for resp in embeddings
               ],
           })

           extraction_context.attrs["corpus"] = corpus

       async def parse_docs_for_structured_data(
           self, doc_infos, out_dir
       ):
           parser = DataCenterParser()

           extraction_context = doc_infos[0].extraction_context
           result_df = await parser.parse(extraction_context)

           if result_df is None or result_df.empty:
               return None

           jurisdiction = extraction_context.attrs.get("jurisdiction")
           result_df["jurisdiction"] = jurisdiction

           return result_df

       @classmethod
       def save_structured_data(cls, doc_infos, out_dir):
           results = []

           for doc_info in doc_infos:
               if doc_info.structured_data is None:
                   continue

               df = doc_info.structured_data
               jurisdiction = df.get("jurisdiction", [None])[0]

               row = {
                   "jurisdiction": jurisdiction,
                   "facility_type": df.get("facility_type", [None])[0],
                   "power_capacity_mw": df.get(
                       "power_capacity", [None]
                   )[0],
                   "cooling_method": df.get(
                       "cooling_system", [None]
                   )[0],
                   "setback_ft": df.get("setback", [None])[0],
                   "noise_limit_db": df.get("noise_limit", [None])[0],
                   "height_limit_ft": df.get("height_limit", [None])[0],
               }

               results.append(row)

           if not results:
               return

           output_df = pd.DataFrame(results)
           output_path = Path(out_dir) / "data_center_ordinances.csv"
           output_df.to_csv(output_path, index=False)

   register_plugin(DataCenterExtractorCustom)

Registration with ``register_plugin()`` makes your class discoverable by
the CLI. Without it, COMPASS cannot find your plugin even if the code is
correct.

Running your custom plugin
^^^^^^^^^^^^^^^^^^^^^^^^^^

The CLI interface is the same as standard plugins. COMPASS does not care
about your internal implementation choices:

.. code-block:: bash

   export OPENAI_API_KEY="your-key"

   compass process --config my_config.yaml

Users select your plugin through the identifier you specify
(``data_center_custom`` in this example). The remaining implementation
details stay internal to the plugin.

When to choose each level
--------------------------

You have seen four levels of customization. Here is when each makes sense.

OrdinanceExtractionPlugin
  Use when your domain fits the standard pattern: keyword filtering works,
  regulations are in a single document or closely related documents,
  extraction is sequential, and standard CSV schema works.

  **Signals:** Simple keyword lists define your target content, each
  feature extracts from a self-contained text section, you need 1-5
  features, output is flat tabular data.

  **Examples:** Battery storage, solar setbacks, wind turbine heights.

FilteredExtractionPlugin with hooks
  Use when standard filtering works but you need to preprocess documents
  or validate filtering results. Hooks give you control at specific points
  without rebuilding the filtering pipeline.

  **Signals:** Document metadata needs extraction, jurisdiction names need
  normalization, you want to log or validate at pipeline stages, most
  filtering logic is standard but small customizations are needed.

  **Examples:** Extracting capacity thresholds from filenames, normalizing
  county names, flagging high-priority documents.

BaseExtractionPlugin with custom filtering
  Use when semantic search outperforms keyword filtering or when you need
  cross-document context. You control the entire filtering pipeline and
  can build sophisticated document corpora.

  **Signals:** Regulations span multiple documents with cross-references,
  keyword patterns are unreliable, you need to query the corpus multiple
  times, filtering is a bottleneck in your pipeline.

  **Examples:** Multi-code ordinances, technical standards with appendices,
  regulations with external references.

BaseExtractionPlugin with RAG extraction
  Use when feature extraction benefits from semantic queries and parallel
  processing. You trade setup complexity for query flexibility and
  performance.

  **Signals:** You are extracting 10+ features, each feature needs
  targeted context queries, relevant text is scattered across many pages,
  sequential parsing is too slow, standard extraction text is too broad.

  **Examples:** Data centers, complex industrial facilities, multi-phase
  development projects, comprehensive use permits.

The decision is not about preference—it is about matching the pattern to
the domain. Start with the simplest pattern that works, then move to more
complex patterns only when simpler ones are insufficient.

Debugging advanced plugins
---------------------------

Custom plugins introduce new failure modes. Here is how to debug them
systematically.

Async task deadlocks
^^^^^^^^^^^^^^^^^^^^^

**Symptom:** Parser hangs indefinitely during ``asyncio.gather()``.

**Check:**

1. Are rate limits blocking progress? Log task start/completion times.
2. Is one task raising unhandled exceptions? Wrap tasks in try-except or
   use ``return_exceptions=True`` in gather.
3. Are you hitting API timeouts? Increase LLM timeout settings.
4. Is semaphore count too low? A single blocked task can deadlock if
   concurrency is 1.

Label chain misalignment
^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom:** Parser receives wrong or missing text.

**Check:**

1. Are you using labels with ``BaseExtractionPlugin``? Labels are for
   ``OrdinanceExtractionPlugin`` only. Custom plugins pass data directly.
2. Did you store corpus in correct location? Use
   ``extraction_context.attrs["corpus"]``.
3. Is parser reading from correct location? Match storage and retrieval
   keys exactly.

Missing structured data
^^^^^^^^^^^^^^^^^^^^^^^

**Symptom:** ``save_structured_data()`` receives empty ``doc_infos``.

**Check:**

1. Is parser returning data? Log return value in
   ``parse_docs_for_structured_data()``.
2. Does parser return DataFrame? Standard pattern expects pandas
   DataFrame, not dict.
3. Are you assigning to ``doc_info.structured_data``? Custom plugins must
   set this attribute explicitly in some cases.
4. Is save logic checking for None correctly? Add defensive None checks.

Comparison with water rights
-----------------------------

The water rights plugin was the inspiration for this guide. Here is how
your data center plugin compares:

Similarities
^^^^^^^^^^^^

- Both inherit from ``BaseExtractionPlugin`` for full control
- Both build embedded corpora in ``filter_docs()``
- Both use RAG with a lightweight vector search helper for extraction
- Both implement parallel async parsing with ``asyncio.gather()``
- Both override ``save_structured_data()`` for custom output

Differences
^^^^^^^^^^^

- Water rights uses Texas groundwater conservation districts as
  jurisdictions; data centers use standard county/city jurisdictions
- Water rights queries corpus generically for all features; data centers
  use feature-specific queries for targeted context
- Water rights has 16+ features; data centers focus on 7 core features
- Water rights outputs a single aggregated CSV; data centers could output
  per-jurisdiction files or JSON

The architectural patterns remain the same even though the domains differ.
``BaseExtractionPlugin`` lets you apply the same approach to a wide range
of regulatory extraction problems.

Where to go next
----------------

You now understand the full spectrum of COMPASS plugin customization. You
have built hooks, custom filters, RAG extractors, and custom outputs. You
can match patterns to domain requirements and debug failures systematically.

Next steps:

- Build your own custom plugin for a domain that needs RAG or custom output
- Examine the water rights implementation at `compass/extraction/water/plugin.py <https://github.com/NatLabRockies/COMPASS/blob/main/compass/extraction/water/plugin.py>`_ for production patterns
- Explore one-shot schema-based plugins at `compass/plugin/one_shot/base.py <https://github.com/NatLabRockies/COMPASS/blob/main/compass/plugin/one_shot/base.py>`_ for rapid prototyping
- Review LLM usage tracking at `compass/services/usage.py <https://github.com/NatLabRockies/COMPASS/blob/main/compass/services/usage.py>`_ to understand cost monitoring

The framework provides controlled extension points at every level. Use the
simplest pattern that works, and move to advanced patterns only when simpler
ones no longer suffice. This approach yields maintainable, performant
extraction pipelines.
