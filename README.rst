*******************************************************************************************
Infrastructure Continuous Ordinance Mapping for Planning and Siting Systems (INFRA-COMPASS)
*******************************************************************************************

|License| |PythonV| |PyPi| |Ruff| |Pixi| |SWR|

.. |PythonV| image:: https://badge.fury.io/py/NREL-COMPASS.svg
    :target: https://pypi.org/project/NREL-COMPASS/

.. |PyPi| image:: https://img.shields.io/pypi/pyversions/NREL-COMPASS.svg
    :target: https://pypi.org/project/NREL-COMPASS/

.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff

.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-orange.svg
    :target: https://opensource.org/licenses/BSD-3-Clause

.. |Pixi| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json
    :target: https://pixi.sh

.. |SWR| image:: https://img.shields.io/badge/SWR--25--62_-blue?label=NREL
    :alt: Static Badge

.. inclusion-intro


INFRA-COMPASS is an innovative software tool that harnesses the power of Large Language Models (LLMs)
to automate the compilation and continued maintenance of an inventory of state and local codes
and ordinances pertaining to energy infrastructure.


Installing INFRA-COMPASS
========================
The quickest way to install INFRA-COMPASS for users is from PyPi:

.. code-block:: bash

    pip install nrel-compass

If you would like to install and run INFRA-COMPASS from source, we recommend using `pixi <https://pixi.sh/latest/>`_:

.. code-block:: bash

    git clone git@github.com:NREL/COMPASS.git; cd COMPASS
    pixi run compass

Before performing any web searches (i.e. running the COMPASS pipeline), you will need to run the following command:

.. code-block:: shell

    rebrowser_playwright install

If you are using ``pixi``, don't forget to prefix this command with ``pixi run`` or initialize a shell using ``pixi shell``.
For detailed instructions, see the `installation documentation <https://nrel.github.io/COMPASS/misc/installation.html>`_.


Quickstart
==========
To run a quick INFRA-COMPASS demo, set up a personal OpenAI API key and run:

.. code-block:: shell

    pixi run openai-solar-demo <your API key>

This will run a full extraction pipeline for two counties using ``gpt-4o-mini`` (costs ~$0.35).
For more information on configuring an INFRA-COMPASS run, see the
`execution basics example <https://nrel.github.io/COMPASS/examples/execution_basics/README.html>`_.


Development
===========
Please see the `Development Guidelines <https://nrel.github.io/COMPASS/dev/index.html>`_
if you wish to contribute code to this repository.
