******************************
Parsing Existing PDF Documents
******************************

If you already have PDF documents you'd like to analyze, there's no need to run the entire web scraping and
ingestion pipeline. This tutorial walks you through how to process those existing PDFs using only the
ordinance extraction portion of the workflow.

We'll focus on using the ``COMPASS`` and ``ELM`` libraries to load a local PDF file, extract ordinance-related text,
and convert it into structured values. By the end, you'll have a fully working script that reads a PDF and outputs
both a text file of the ordinance and a structured CSV.

If you're only looking to download documents without extracting content, you may want to look at the
`ELM tutorial <https://github.com/NREL/elm/blob/main/examples/web_scraping_pipeline/example_scrape_wiki.ipynb>`_,
which focuses on scraping and saving documents alone.


Extraction Components
=====================

The ordinance extraction pipeline is composed of several components, which are visualized in the diagram below:

.. mermaid::

    flowchart LR
        A -->|Ordinance Content Checker<br>&#40<span style='color: #c792ea'><code>check_for_ordinance_info</code></span>&#41| B
        B -->|Ordinance Text Extractor<br>&#40<span style='color: #c792ea'><code>extract_ordinance_text_with_llm</code></span>&#41| C
        C -->|Ordinance Extractor<br>&#40<span style='color: #c792ea'><code>extract_ordinance_values</code></span>&#41| D
        A@{ shape: lined-document, label: "Ordinance Document<br>&#40<span style='color: #a9c77d'><code>PDFDocument</code></span>&#41" }
        B@{ shape: docs, label: "Ordinance Text Chunks<br>&#40<span style='color: #a9c77d'><code>str</code></span>&#41"}
        C@{ shape: doc, label: "Ordinance Text<br>&#40<span style='color: #a9c77d'><code>str</code></span>&#41"}
        D@{ shape: lin-cyl, label: "Structured Ordinances<br>&#40<span style='color: #a9c77d'><code>DataFrame/CSV</code></span>&#41" }


In the following sections, we will discuss how to invoke each of these components and then
put them together in a short script to perform ordinance extraction on a local PDF document.


Document Class
--------------
The first thing we need is a way to represent the PDF we're working with. In this workflow, documents are wrapped
using the :class:`elm.web.document.PDFDocument` class (or the corresponding :class:`elm.web.document.HTMLDocument`
class for HTML content stored as text). This class loads the raw text from a file and includes helpful processing
routines (e.g., removing headers). This class also tracks metadata associated with the document metadata in the
``doc.attrs`` attribute. This attribute helps ``COMPASS`` track metadata such as the source URL, document date,
ordinance text, and extracted structured values. Many of the ``COMPASS`` functions require a document as an input.

To load a local file, simply use:

.. code-block:: python

    from elm.web.document import PDFDocument

    doc = PDFDocument.from_file("Decatur County, Indiana.pdf")

At this stage, the document's ``attrs`` dictionary is still empty. As we go through the steps below, that dictionary
will be filled with the output of each stage in the extraction process.


Setting Up the LLM Configuration
--------------------------------
Next, we'll configure how we want to interact with an OpenAI large language model (LLM). This is done using the
:class:`~compass.llm.config.OpenAIConfig` class. Let's take a look at a basic configuration:

.. code-block:: python

    from compass.llm import OpenAIConfig

    caller_args = OpenAIConfig(
        name="gpt-4o-mini",
        llm_call_kwargs={"temperature": 0},
        llm_service_rate_limit=500_000,
        text_splitter_chunk_size=10_000,
        text_splitter_chunk_overlap=500,
        client_type="azure",
        client_kwargs={
            "api_key": "<your API key>",
            "api_version": "<your API version>",
            "azure_endpoint": "<your API endpoint>",
        },
    )


This object specifies the model we'll use (in this case, ``gpt-4o-mini``), the rate limit for this model
(500k tokens per minute), what parameters to use for the LLM queries (e.g. ``"temperature": 0``), how to
split text into manageable chunks (10k tokens per chunk with a 500 token overlap), and how to authenticate
with Azure OpenAI. While API credentials can be included inline like this, it's much safer and more flexible
to use environment variables instead. We'll revisit this in the :ref:`final execution step<running>`.

You can create multiple ``OpenAIConfig`` objects if you need different models for different stages of extraction,
but in this tutorial, we'll stick to a single configuration for simplicity.


COMPASS Services
----------------
``COMPASS`` Services are utility classes designed to run asynchronously alongside your script. You can
invoke these services using class methods, allowing you to access them anywhere in your code without
needing to pass around object instances. Internally, each service maintains a queue of processing "requests,"
which are handled as soon as the necessary resources become available (e.g., when the process is within the
rate limit for LLM queries).

To use these services, you must ensure that their queue is properly initialized and that each service
is actively monitoring the queue and ready to process requests. In practice, this requires using the
:class:`~compass.services.provider.RunningAsyncServices` context manager, as demonstrated below:

.. code-block:: python

    from compass.services.provider import RunningAsyncServices

    services = [...]  # Define your services here
    async with RunningAsyncServices(services):
        # Your extraction steps go here

Under the hood, this starts a queue manager that feeds tasks to the appropriate services. Each service runs in parallel,
ensuring that multiple documents (or stages) can be handled efficiently.


Checking for Ordinance Content
------------------------------
The first functional step in extraction is to determine whether the document contains any ordinance-related information.
Even if you're fairly sure that it does, this step is essential because it saves the text chunks from the document that
contain ordinance information, enabling the next step — ordinance text extraction.

To do this, we'll use the :func:`~compass.extraction.apply.check_for_ordinance_info` function. This function uses a
combination of keyword heuristics and LLM evaluation to identify ordinance content and collect it into a new field
in the document. Here's how that might look:

.. code-block:: python

    from compass.extraction.apply import check_for_ordinance_info
    from compass.extraction.solar import SolarHeuristic, SolarOrdinanceTextCollector

    doc = await check_for_ordinance_info(
        doc,
        llm_callers=caller_args,
        heuristic=SolarHeuristic(),
        ordinance_text_collector_class=SolarOrdinanceTextCollector,
        permitted_use_text_collector_class=None,
    )


What this function does is scan the text for solar-related ordinance language. If it finds any, it stores the relevant
(concatenated) chunks in ``doc.attrs["ordinance_text"]``. To call this function, we passed in the document along with
the LLM calling arguments that we set up in `Setting Up the LLM Caller`_. We also specified the use of the
:class:`~compass.extraction.solar.ordinance.SolarHeuristic`, which helps reduce LLM costs by applying a simple
keyword-based heuristic to each document chunk before sending it to the LLM. Finally, we indicated that the
:class:`~compass.extraction.solar.ordinance.SolarOrdinanceTextCollector` class should be used to search for solar ordinance
text in the document — rather than, for example, the
:class:`~compass.extraction.wind.ordinance.WindOrdinanceTextCollector`, which would look for wind ordinance text instead.

You can also enable permitted-use extraction by specifying a permitted use collector class (e.g.
:class:`~compass.extraction.solar.ordinance.SolarPermittedUseDistrictsTextCollector`) for the
``permitted_use_text_collector_class`` parameter. For this tutorial, we'll keep things focused on ordinances, so
we have left that output as ``None``.


Isolating the Ordinance Text
----------------------------
Once we've located the general sections where the ordinances are mentioned, we'll want to refine the text further.
The identified chunks are often too broad to use directly in downstream processing, so we'll pass them through another
LLM-powered step that filters the content to only the most relevant ordinance language.

We'll do that using the :func:`~compass.extraction.apply.extract_ordinance_text_with_llm` function:

.. code-block:: python

    from compass.llm import LLMCaller
    from compass.extraction.apply import extract_ordinance_text_with_llm
    from compass.extraction.solar import SolarOrdinanceTextExtractor

    doc, ord_text_key = await extract_ordinance_text_with_llm(
        doc,
        caller_args.text_splitter,
        extractor=SolarOrdinanceTextExtractor(
            LLMCaller(llm_service=caller_args.llm_service)
        ),
        original_text_key="ordinance_text",
    )

This step reads the raw text chunks stored in ``doc.attrs["ordinance_text"]`` and returns a more focused subset — just
the ordinance language itself. The first argument to this function is the ordinance document, which must contain an
``"ordinance_text"`` key in its ``doc.attrs`` dictionary. This key holds the concatenated text chunks identified as
likely containing ordinance information. It's automatically added for us by the
:func:`~compass.extraction.apply.check_for_ordinance_info` function — assuming ordinance text is present.

Next, we pass in the text splitter instance, which will be used to divide the concatenated text into smaller chunks.
We also provide a :class:`~compass.extraction.solar.ordinance.SolarOrdinanceTextExtractor` instance, which performs the
actual ordinance text extraction.

Finally, we specify that the concatenated text is located under the ``"ordinance_text"`` key in ``doc.attrs``. If the extraction
is successful, the resulting ordinance text is stored back in the ``doc.attrs`` dictionary under the key specified by
``ord_text_key`` (which depends on the extractor instance being used). We'll use that key in the next step.


Extracting Structured Values
----------------------------
With the ordinance language in hand, we're now ready to extract structured ordinance values — things like setback requirements,
noise restrictions, or installation constraints. This is the final step in the pipeline.

We'll use the :func:`~compass.extraction.apply.extract_ordinance_values` function to convert natural language into
structured data:

.. code-block:: python

    from compass.extraction.apply import extract_ordinance_values
    from compass.extraction.solar import StructuredSolarOrdinanceParser

    doc = await extract_ordinance_values(
        doc,
        parser=StructuredSolarOrdinanceParser(
            llm_service=caller_args.llm_service
        ),
        text_key=ord_text_key,
        out_key="ordinance_values",
    )


The first argument to this function is the ordinance document, which must include the ``ord_text_key`` in its
``doc.attrs`` dictionary. This key contains the ordinance text that will be used as context for running decision
trees to extract structured values. (The specific value of this key depends on the text extractor used in
`Isolating the Ordinance Text`_.)

Next, we provide a :class:`~compass.extraction.solar.parse.StructuredSolarOrdinanceParser` instance. This class sets up
and runs the decision trees, returning the extracted results in a structured format — typically as CSV.

Finally, we specify which keys in the ``doc.attrs`` dictionary should be used to store the input ordinance text
and the output structured data. When this function completes, you'll have a list of parsed values stored in
``doc.attrs["ordinance_values"]``, ready to be written to a CSV or used in a downstream application.


Putting It All Together
=======================
To wrap everything into a working script, we combine the document loading, LLM configuration, and the extraction steps into
one file. We also include additional logging setup so you can see the progress in your terminal. Here's the full example script:

.. literalinclude:: parse_pdf.py
    :language: python

You can also view this script in the repository:
`parse_pdf.py <https://github.com/NREL/COMPASS/blob/main/examples/parse_existing_docs/parse_pdf.py>`_

.. _running:

Running the Script
==================
Before executing the script, you'll need to define some environment variables that allow it to authenticate with Azure OpenAI:

.. code-block:: shell

    export AZURE_OPENAI_API_KEY=<your API key>
    export AZURE_OPENAI_ENDPOINT=<your API endpoint>
    export AZURE_OPENAI_VERSION=<your API version>

Alternatively, you can hardcode them in the script, but environment variables are the preferred option for both security and
portability.

You may also need to change the model name in the script to match the model name of your deployment. Once everything is set,
run the script from your terminal:

.. code-block:: shell

    python parse_pdf.py

You should see logs as each stage completes. When the script completes, you'll see two new files:

- ``Decatur County, Indiana Ordinance Text.txt``: file containing the extracted ordinance language.
- ``Decatur County, Indiana Ordinances.csv``: file with the structured ordinance values.

With ~100 lines of code, you can use ``COMPASS`` and ``ELM`` to extract structured ordinance data from existing PDFs.
This approach avoids the overhead of a full scraping pipeline and makes it easy to apply ordinance extraction to any
local documents you already have on hand.
