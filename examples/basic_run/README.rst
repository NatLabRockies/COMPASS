*********************
Running INFRA-COMPASS
*********************

This example walks you through setting up and executing your first INFRA-COMPASS run.


Prerequisites
=============
We recommend enabling Optical Character Recognition (OCR) for PDF parsing, which
allows the program to process scanned documents. To enable OCR, you'll need to install ``pytesseract``.
If you installed COMPASS via PyPI, you may need to install a few additional dependencies:

.. code-block:: shell

    $ pip install pytesseract pdf2image

If you're using ``pixi`` to run the pipeline (recommended), these libraries are included by default.

In either case, you may still need to complete a few additional setup steps if this is your first time
installing Google's ``tesseract`` utility. Follow the installation instructions
`here <https://pypi.org/project/pytesseract/#:~:text=lang%5D%20image_file-,INSTALLATION,-Prerequisites%3A>`_.


Setting Up the Run Configuration
================================
The INFRA-COMPASS configuration file—written in either ``JSON`` or ``JSON5`` format—is a simple config that
defines parameters for running the process. Each key in the config corresponds to an argument for the function
`process_counties_with_openai <https://nrel.github.io/COMPASS/_autosummary/compass.scripts.process.process_counties_with_openai.html#compass.scripts.process.process_counties_with_openai>`_.
Refer to the linked documentation for detailed and up-to-date descriptions of each input.


Minimal Config
--------------
At a minimum, the INFRA-COMPASS config file requires three keys: ``"out_dir"``, ``"jurisdiction_fp"``, and ``"tech"``.

- ``out_dir``: Path to the output directory. Will be created if it does not exist.
- ``jurisdiction_fp``: Path to a CSV file containing ``County`` and ``State`` columns. Each row defines a jurisdiction to process. See the `example CSV <https://github.com/NREL/elm/blob/main/examples/basic_run/jurisdictions.csv>`_.
- ``tech``: A string representing the infrastructure or technology focus for the run.

In `config_bare_minimum.json5 <https://github.com/NREL/COMPASS/blob/main/examples/basic_run/config_bare_minimum.json5>`_,
we show a minimal working configuration that includes only the required keys.

.. literalinclude:: config_bare_minimum.json5
    :language: json5

This configuration is sufficient for a basic run using default settings and assumes the following:

**Environment Configuration**

Your LLM credentials and endpoints should be configured as environment variables. For example, when using Azure OpenAI:

- ``AZURE_OPENAI_API_KEY``
- ``AZURE_OPENAI_VERSION``
- ``AZURE_OPENAI_ENDPOINT``

**LLM Model Defaults**

This minimal setup uses the default LLM model for INFRA-COMPASS — ``gpt-4o`` as of April 11, 2025.
To override this default, add a ``model`` key to your config:

.. code-block:: json

    "model": "gpt-4o-mini"


Typical Config
--------------
In most cases, you'll want more control over the execution parameters, especially those related to the LLM configuration.
You can review all available inputs in the
`process_counties_with_openai <https://nrel.github.io/COMPASS/_autosummary/compass.scripts.process.process_counties_with_openai.html#compass.scripts.process.process_counties_with_openai>`_
documentation.
In `config_recommended.json5 <https://github.com/NREL/COMPASS/blob/main/examples/basic_run/config_recommended.json5>`_, we
demonstrate a typical configuration that balances simplicity with additional control over execution parameters.

.. literalinclude:: config_recommended.json5
    :language: json5

This setup supports most users' needs while providing flexibility for key configurations:

**LLM Configuration**
Customize the LLM behavior and performance using:

- ``llm_call_kwargs``: Sets LLM-specific query parameters like ``temperature`` and ``timeout``.
- ``llm_service_rate_limit``: Controls how many tokens can be processed per minute. Set this as high as your deployment will allow to speed up processing.
- ``text_splitter_chunk_size`` and ``text_splitter_chunk_overlap``: Controls how large each text chunk sent to the model is.

.. WARNING::

   Be cautious when adjusting the ``"text_splitter_chunk_size"``. Larger chunk sizes increase token usage, which may result in higher costs per query.

**LLM Credentials**
You can also specify LLM credentials and endpoint details directly in the config under the ``client_kwargs`` key.
Note that while this can be convenient for quick testing, storing credentials in plaintext is not recommended for production environments.

**SSL Configuration**
Set ``verify_ssl`` to ``false`` in ``file_loader_kwargs`` to bypass certificate verification errors, especially useful when running behind the NREL VPN.
If you're not using the VPN, it's best to leave this value as the default (``true``).

**OCR Integration**
As noted in the `Prerequisites`_ section, we recommend enabling OCR using ``pytesseract``. To enable OCR for scanned PDFs,
you must provide the path to the ``tesseract`` executable using the ``pytesseract_exe_fp`` input.
You can locate the executable path by running:

.. code-block:: shell

    $ which tesseract

Omit the ``pytesseract_exe_fp`` key to disable OCR functionality.


Kitchen Sink Config
-------------------

In `config_kitchen_sink.json5 <https://github.com/NREL/COMPASS/blob/main/examples/basic_run/config_recommended.json5>`_,
we show what a configuration might look like that utilizes all available parameters.

.. literalinclude:: config_kitchen_sink.json5
    :language: json5

This setup provides maximum flexibility and is suitable for power users who need fine-grained control
over processing behavior, model assignment, cost monitoring, and concurrency. Below are descriptions of
the most notable components:

**Multiple Model Definitions**
You can specify multiple LLMs under the ``"model"`` key, each with a unique name and a list of associated tasks.
Every task must be handled by exactly one model, and one of the entries must have ``"tasks": "default"`` to catch anything unspecified.
The full list of assignable tasks are found as ``Attributes`` of the :class:`~compass.utilities.enums.LLMTasks` enum.

**LLM Configuration**
Each model includes:

- ``llm_call_kwargs``: Sets LLM-specific query parameters like ``temperature`` or ``timeout``.
- ``llm_service_rate_limit``: Controls how many tokens can be processed per minute (useful for avoiding rate limit errors from the LLM provider). Set this as high as your deployment will allow to speed up processing.
- ``text_splitter_chunk_size`` / ``text_splitter_chunk_overlap``: Controls how large each text chunk sent to the model is. Larger chunks increase context at the cost of higher token usage.
- ``client_type``: Specifies the API provider (e.g., ``"azure"`` or ``"openai"``).
- ``client_kwargs``: Holds credentials and endpoint configuration for the model client, if not specified using environment variables.

**Concurrency Settings**
The following settings allow tuning for system resource usage and rate limits:

- ``max_num_concurrent_browsers``: Limits the number of headless browsers launched for document discovery.
- ``max_num_concurrent_jurisdictions``: Controls how many jurisdictions are processed in parallel.

**OCR Integration**
Set the ``pytesseract_exe_fp`` key to enable OCR support for scanned PDFs. Omit this key if OCR is not needed.

**Directories**
Several directory names are configurable to manage where outputs and intermediate files are stored:

- ``out_dir``: Final results and processed outputs.
- ``log_dir``: Execution logs.
- ``clean_dir``: Cleaned text files.
- ``ordinance_file_dir``: Raw ordinance documents.
- ``jurisdiction_dbs_dir``: Internal DBs for tracking progress.

.. NOTE:: Be sure to provide full paths to all files/directories unless you are executing the program from your working folder.


**LLM Cost Reporting**
The ``llm_costs`` block provides per-million-token pricing for each model.
This allows the script to display real-time and final cost estimates based on tracked usage.
Any model not found in the ``llm_costs`` block will not contribute to the final cost estimate.



Execution
=========
Once you are happy with the configuration parameters, you can kick off the processing using

.. code-block:: shell

    $ compass process -c config.json

If you're using ``pixi``, activate the shell first:

.. code-block:: shell

    $ pixi shell
    $ compass process -c config.json5

or run with pixi directly:

.. code-block:: shell

    $ pixi run compass process -c config.json5

Replace ``config.json5`` with the path to your actual configuration file.

You may also wish to add a ``-v`` option to print logs to the terminal (however, keep in mind that the code runs
asynchronously, so the the logs will not print in order).

During execution, INFRA-COMPASS will:
1. Load and validate the jurisdiction CSV.
2. Attempt to locate and download relevant ordinance documents for each jurisdiction.
3. Parse and validate the documents.
4. Extract relevant ordinance text from the documents.
5. Parse the extracted text to determine the quantitative and qualitative ordinance values within, using decision tree-based LLM queries.
6. Output structured results to your configured ``out_dir``.

The runtime duration varies depending on the number of jurisdictions, the number of documents found for each jurisdiction,
and the rate limit/output token rate of the LLM(s) used.


Outputs
=======

After completion, you'll find several outputs in the ``out_dir``:

- **Extracted Ordinances**: Structured CSV files containing parsed ordinance values.
- **Ordinance Documents**: PDF or text (HTML) documents containing the legal ordinance.
- **Cleaned Text Files**: Text files containing the ordinance-specific text excerpts portions of the downloaded documents.
- **Metadata Files**: JSON files describing metadata parameters corresponding to your run.
- **Logs and Debug Files**: Helpful for reviewing LLM prompts and tracing any issues.

You can now use these outputs for downstream analysis, visualization, or integration with other NREL tools like
`reVX setbacks <https://nrel.github.io/reVX/misc/examples.setbacks.html>`_.
