.. raw:: html

    <p align="center">
        <img src="docs/source/_static/logo_horiz.png" />
    </p>

---------

.. *******************************************************************************************
.. Infrastructure Continuous Ordinance Mapping for Planning and Siting Systems (INFRA-COMPASS)
.. *******************************************************************************************

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


What is INFRA-COMPASS?
======================
INFRA-COMPASS is an innovative software tool that harnesses the power of Large Language
Models (LLMs) to automate the compilation and continued maintenance of an inventory of
state and local codes and ordinances pertaining to energy infrastructure.

At a high level, INFRA-COMPASS does two things: it *retrieves* the right ordinance
documents for each jurisdiction you ask about, and then *extracts* structured data from
those documents into a versioned database that downstream users can query as a CSV, Excel
workbook, or GeoPackage.

.. raw:: html

   <p align="center"><img
     src="https://raw.githubusercontent.com/NatLabRockies/COMPASS/main/docs/source/_static/overview.png"
     alt="High-level overview of the INFRA-COMPASS pipeline: a user defines jurisdictions, INFRA-COMPASS performs document retrieval (web search, document validation) and ordinance extraction (text extraction, value extraction), producing a versioned ordinance database that users can consume as CSV, XLSX, or GeoPackage."
   /></p>

What makes INFRA-COMPASS different from simply asking ChatGPT for ordinance data is the
architecture around the LLM call:

- **Structured, downstream-ready output** — consistent CSV rows with stable column
  names, units, and feature labels that drop straight into siting and capacity-modeling
  tools like `reV <https://github.com/NREL/reV>`_, GIS workflows, or any pipeline that
  needs setbacks, height limits, and noise thresholds as numbers rather than prose.
- **Hallucination guardrails** — cleaned text is checked against the source and dropped
  if it drifts too far, so fabricated values never reach the database.
- **Source-URL traceability** — every record carries a URL back to the original
  ordinance document, so any value can be audited or spot-checked.
- **Cost control** — cheap heuristic filters reject obviously irrelevant text before
  any LLM call runs, making it tractable to extract data across hundreds of
  jurisdictions.


Read more about the tool in the `documentation <https://natlabrockies.github.io/COMPASS/misc/about.html>`_.


Where is the extracted ordinance data?
======================================
The National Laboratories of the Rockies (NLR) typically runs the INFRA-COMPASS pipeline
annually and publishes refreshed datasets to OpenEI. The latest published ordinance datasets
are available here:

- Solar: https://data.openei.org/submissions/8519
- Wind: https://data.openei.org/submissions/8602


Installing INFRA-COMPASS
========================
The quickest way to install INFRA-COMPASS for users is from PyPi:

.. code-block:: bash

    pip install infra-compass

If you would like to install and run INFRA-COMPASS from source, we recommend using `pixi <https://pixi.sh/latest/>`_:

.. code-block:: bash

    git clone git@github.com:NatLabRockies/COMPASS.git; cd COMPASS
    pixi run compass

For detailed instructions and troubleshooting, see the `installation documentation <https://natlabrockies.github.io/COMPASS/misc/installation.html>`_.


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
