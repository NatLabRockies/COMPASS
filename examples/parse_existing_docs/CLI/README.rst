*********************************
Parsing Existing Docs via the CLI
*********************************

If you already have documents that you want to run data extraction on,
you can skip web search and run COMPASS directly against local files.
This example shows the minimal CLI setup for processing local documents.

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
need to generate a mapping of jurisdiction codes to lists of document
metadata dicts, where each dict contains (at minimum) a required
``source_fp`` key that points to the local file path.

For example, a minimal local document specification would look like this:

.. literalinclude:: local_docs_minimal.json5
    :language: json5

This mapping can be saved as a config file using any of the formats
supported by COMPASS (JSON, JSON5, YAML, or TOML).

Since we didn't include any additional metadata beyond the required
``source_fp``, COMPASS will perform all of the same document processing
steps that a document retrieved via search would go through, including
legal text validation and date extraction. To skip some or all of these
steps, you can include additional metadata fields in the document dicts
as described in the
`COMPASS documentation <https://natlabrockies.github.io/COMPASS/_autosummary/compass.scripts.process.process_jurisdictions_with_openai.html#compass.scripts.process.process_jurisdictions_with_openai>`_.
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


Running COMPASS
===============
Once everything is configured, you can execute a model run as described in the
`COMPASS Execution Basics <https://natlabrockies.github.io/COMPASS/examples/execution_basics/README.html>`_:

.. code-block:: shell

    compass process -c config.json5

If you are using ``pixi``:

.. code-block:: shell

    pixi run compass process -c config.json5

Outputs are written under ``./outputs`` by default.
