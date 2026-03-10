*************************************************
INFRA-COMPASS Texas Water Rights One-Shot Example
*************************************************

This directory contains an example configuration for extracting groundwater rights
for several districts in Texas using a one-shot plugin config. To execute this run,
fill out the config file with the appropriate paths and API keys,
then run the following command:

.. code-block:: shell

    compass process -c config.json5 -p plugin_config.yaml


Note that the one-shot plugin will still run location and document type validation,
which may not be desirable in this case. To disable this validation, you would need to
implement your own plugin and manually disable the validation by setting the appropriate
document attributes after collection.
