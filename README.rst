*******************************************************************************************
Infrastructure Continuous Ordinance Mapping for Planning and Siting Systems (INFRA-COMPASS)
*******************************************************************************************

|Ruff| |Pixi| |License| |SWR|

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

.. code-block:: shell

    pip install nrel-compass

If you would like to install INFRA-COMPASS from source, we recommend using `pixi <https://pixi.sh/latest/>`_:

.. code-block:: shell

    git clone git@github.com:NREL/COMPASS.git
    cd COMPASS
    pixi shell

For detailed instructions, see the `installation documentation <https://nrel.github.io/compass/misc/installation.html>`_.


Development
===========
Please see the `Development Guidelines <https://nrel.github.io/compass/dev/index.html>`_
if you wish to contribute code to this repository.
