*********************************
Parsing Existing Docs via the CLI
*********************************

If you already have documents that you want to run data extraction on,
you can skip web search and run COMPASS directly against local files.
This example shows the minimal CLI setup for processing local documents.
It also covers the split ``collect`` and ``extract`` workflow for cases where
you want to persist a local corpus and rerun extraction later.

Prerequisites
=============
Be sure to go over the
`COMPASS Execution Basics <https://natlabrockies.github.io/COMPASS/examples/execution_basics/README.html>`_
to understand how to set up a run environment and model run configuration.
You will be re-using the same execution pattern here with an added input to
point COMPASS to your local files.

Compile Document Info
=====================
The key to running COMPASS against local files is compiling information
about the local documents that we can point COMPASS to. To do this, we
need to generate a mapping of jurisdiction codes (e.g. FIPS codes) to
lists of document metadata dicts, where each dict contains (at minimum)
a required ``source_fp`` key that points to the local file path.

For example, a minimal local document specification would look like this:

.. literalinclude:: local_docs_minimal.json5
    :language: json5

This mapping can be saved as a config file using any of the formats
supported by COMPASS (JSON, JSON5, YAML, or TOML).

If you need to look up the jurisdiction codes to use in the mapping,
you can take a look at the
`list of known jurisdictions <https://github.com/NatLabRockies/COMPASS/blob/main/compass/data/conus_jurisdictions.csv>`_
in the COMPASS repository.

Since we didn't include any additional metadata beyond the required
``source_fp``, COMPASS will perform all of the same document processing
steps that a document retrieved via search would go through, including
legal text validation and date extraction. To skip some or all of these
steps, you can include additional metadata fields in the document dicts
as described in the
`COMPASS documentation <https://natlabrockies.github.io/COMPASS/_autosummary/compass.pipeline.data_classes.ProcessRequest.html#compass.pipeline.data_classes.ProcessRequest>`_.
Below is an example of a more fully specified document mapping that
includes multiple documents, each with additional metadata fields to
skip certain processing steps:

.. literalinclude:: local_docs.json5
    :language: json5


Updating COMPASS Run Config
===========================
Once the local document mapping is compiled, you can point COMPASS to it via
the main run config. You will also need to disable search so that COMPASS
doesn't attempt to retrieve documents from the web in addition to processing
your local files. The rest of the config can be set up as a typical COMPASS
run config with out_dir, tech, and any other relevant settings. Below is a
simple example:

.. literalinclude:: config.json5
    :language: json5

.. NOTE::
    If you are not sure whether your local docs contain the relevant information
    to be extracted, you can leave the web search enabled and COMPASS will
    default back to a web search if no structured data is extracted from the
    local documents.

Of course, your jurisdiction CSV should still be set up to match the jurisdictions
you would like to process:

.. literalinclude:: jurisdictions.csv
    :language: text

In this way, you can build up a corpus of local docs, point your config to the
document mapping, and only ever process the jurisdiction(s) you are interested in.


Choosing the CLI Flow
=====================
If your local document set is ready and you want the original one-command
workflow, you can still use ``compass process``. That remains the simplest
option for local files and does not require any web retrieval when
``perform_se_search`` and ``perform_website_search`` are disabled.

If you want to separate deterministic document collection from LLM-backed
extraction, run the two phases independently:

.. code-block:: shell

    compass collect -c config.json5
    compass extract -c extract_config.json5

The collection step writes ``collection_manifest.json`` into the configured
``out_dir``. For local documents, that manifest records the jurisdiction
metadata plus the persisted parsed text and source-file artifacts needed to
reconstruct the extraction inputs. The extraction config can then point to the
saved manifest with ``collection_manifest_fp``.

Because the documents are already known in this workflow, collection can stay
fully deterministic and does not require any LLM calls.


Running COMPASS
===============
Once everything is configured, you can execute a model run as described in the
`COMPASS Execution Basics <https://natlabrockies.github.io/COMPASS/examples/execution_basics/README.html>`_:

.. code-block:: shell

    compass process -c config.json5

If you are using ``pixi``:

.. code-block:: shell

    pixi run compass process -c config.json5

To run the split workflow with ``pixi``:

.. code-block:: shell

    pixi run compass collect -c config.json5
    pixi run compass extract -c extract_config.json5

Outputs are written under ``./outputs`` by default.
