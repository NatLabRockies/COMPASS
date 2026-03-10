**************************************************
INFRA-COMPASS Texas Water Rights RAG-Based Example
**************************************************

This directory contains an example configuration for extracting groundwater rights
for several districts in Texas. To execute this run, fill out the config file with
the appropriate paths and API keys, then run the following command:

.. code-block:: shell

    export OPENAI_API_KEY="dummy";
    export AZURE_OPENAI_KEY="<your EMBEDDING API key>";
    export AZURE_OPENAI_API_KEY="<your EMBEDDING API key>";
    export AZURE_OPENAI_VERSION="<your EMBEDDING API version>";
    export AZURE_OPENAI_ENDPOINT="<your EMBEDDING API endpoint>";

    compass process -c config.json5
