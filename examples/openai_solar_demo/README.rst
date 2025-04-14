**********************
INFRA-COMPASS Demo Run
**********************

This directory contains an example configuration intended to get you up-and-running as quickly as possible.
To execute this run, set up your own OpenAI key (make sure you have at least $0.50 or so available for API
calls), and then tun the following command:

.. code-block:: shell

    export OPENAI_API_KEY="<your API key>"; compass process -c config.json5

Alternatively, you can run this demo using ``pixi``:

.. code-block:: shell

    pixi run openai-solar-demo <your API key>
