*******************
One-Shot Extraction
*******************

This example shows how to author a one-shot extraction schema and run it
through COMPASS. The one-shot plugin uses your schema to extract structured
data in a single LLM call.


Prerequisites
=============
Be sure to go over the
`COMPASS Execution Basics <https://natlabrockies.github.io/COMPASS/examples/execution_basics/README.html>`_
to understand how to set up a run environment and model run configuration.
Once your one-shot schema is established, you will be executing the data
extraction pipeline in the same manner as described in that example.


Create Your Schema
==================
To start off, you will need to create a one-shot JSON schema that describes the
extraction output shape and embeds the extraction logic in schema field
descriptions. The easiest way to do this is by copying
`wind_schema.json <https://github.com/NatLabRockies/COMPASS/blob/main/examples/one_shot_schema_extraction/wind_schema.json>`_
and adjusting it for your domain.

At a minimum, the schema must return an object with an ``outputs`` array, where
each item is an extraction record with the required fields shown below:

.. code-block:: json

    {
        "type": "object",
        "required": ["outputs"],
        "properties": {
            "outputs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "feature",
                        "value",
                        "units",
                        "section",
                        "summary"
                    ]
                }
            }
        }
    }

The main field here is ``feature``, which is the ID of the extracted feature
(e.g., a setback distance or a maximum allowed height). The other fields
(``value``, ``units``, ``section``, and ``summary``) are important for keeping
the output consistent across various extractions and allowing a central database
to keep track of the scraped data.

Once the schema for the ``outputs`` array is finalized, you can add additional
keys starting with a ``$`` to encode instructions, examples, and edge case
handling logic that the model can refer to when parsing the text. These extra
keys are not required, and they are ignored for the purposes of creating the
structure of the outputs themselves, but they often provide crucial context
that improves extraction accuracy.

For example, the
`wind extraction schema <https://github.com/NatLabRockies/COMPASS/blob/main/examples/one_shot_schema_extraction/wind_schema.json>`_
contains a ``$definitions`` key with detailed instructions on how to interpret
setback multipliers and how to choose the most restrictive value when multiple
setback distances are given in the text. This is reminiscent of the "decision logic"
that you would normally encode in a decision tree for a traditional plugin,
but here the logic is embedded in the schema itself and interpreted by the model
at extraction time. This approach allows you to encode complex edge case handling
logic without having to write any code, and it also allows you to easily update
the logic by simply editing the schema.

The schema also includes a ``$examples`` key with example extractions that the model
can refer to when deciding how to parse the text. You can be as detailed as you want
in these instructions, and you can experiment with different outputs to tune the
model's understanding of the task and the desired output format.

Finally, the same schema includes a ``$instructions`` key with general instructions
for the model to follow when parsing the text. This is a good place to reinforce the
importance of following the schema and to provide any additional context that might be
helpful for the model to know when performing the extraction.

You can add/remove as many of these extra keys as you want, and you can experiment with
different ways of encoding the instructions and examples to see what works best for your
particular use case. The main thing to keep in mind is that the core structure of the
output must be defined by the ``outputs`` array in the schema, and any additional context
or instructions should be provided through these extra keys.

.. NOTE:: You can compare the `one-shot wind schema <https://github.com/NatLabRockies/COMPASS/blob/main/examples/one_shot_schema_extraction/wind_schema.json>`_
   to the existing decision trees in the `wind energy plugin <https://github.com/NatLabRockies/COMPASS/tree/main/compass/extraction/wind>`_
   to get a feel for the translation of decision tree logic to schema descriptions.


.. Important Schema Components
.. ---------------------------
.. **Feature Catalog**
.. Define the allowed feature IDs (often as an enum) under
.. ``outputs.items.properties.feature``. These IDs are what the parser uses to
.. create the final output rows.

.. **Field Requirements**
.. Enforce ``required`` fields and ``additionalProperties: false`` to keep the
.. output consistent. The core fields are ``feature``, ``value``, ``units``,
.. ``section``, and ``summary``.

.. **Decision Logic in Descriptions**
.. Use field descriptions and ``$definitions`` to encode extraction rules and
.. edge cases (e.g., how to choose the most restrictive value or how to interpret
.. setback multipliers).

.. **Instructions and Examples**
.. Use ``$instructions`` and ``$examples`` to reinforce the desired output and to
.. anchor the model on your conventions.


Build a Plugin Config
=====================
Once you have defined your schema, the hard work is done! The next step is to
build a one-shot plugin config that tells COMPASS how to use the schema and
how to retrieve and filter documents. As with all configs in COMPASS, you may
define your plugin configuration via JSON, JSON5, YAML, or TOML.

At a minimum, you must supply a ``schema`` key (either a dictionary containing the
full schema or a path to a schema file):

.. literalinclude:: plugin_config_minimal.json
    :language: json


If you want a little bit more control over the extraction pipeline, you may
specify several additional keys that let you customize query templates, website
filters, and text extraction prompts:


.. literalinclude:: plugin_config_simple.json5
    :language: json5

The key options are listed below:

- ``data_type_short_desc``: Short label used in prompts (e.g., ``wind energy ordinance``).
- ``query_templates``: Search queries with a ``{jurisdiction}`` placeholder.
- ``website_keywords``: Keyword weights for document search prioritization.
- ``collection_prompts``: Prompt list for chunk filtering, or ``true`` to auto-generate.
- ``text_extraction_prompts``: Prompt list for text consolidation, or ``true`` to auto-generate.
- ``cache_query_templates``: Cache generated query templates and keywords. By default, ``true``.
- ``extraction_system_prompt``: Optional system prompt override for extraction.


See `this documentation <https://natlabrockies.github.io/COMPASS/_autosummary//compass.plugin.one_shot.base.create_schema_based_one_shot_extraction_plugin.html#compass.plugin.one_shot.base.create_schema_based_one_shot_extraction_plugin>`_
for further details.

If you want full control over all of the above options, you can specify them directly in the config
as shown below. Note that you can also specify custom prompts for the collection and text extraction steps,
which will give you even more control over the extraction pipeline and allow you to further tune the
performance of the model.


.. literalinclude:: plugin_config.yaml
    :language: yaml


Execution
=========
Once both the schema and plugin configuration are set up, you can run your newly created
one-shot plugin alongside the standard COMPASS pipeline using the ``--plugin`` flag.
The main run config still controls core pipeline settings and must include a ``tech``
value that matches your target technology.

.. code-block:: shell

    compass process -c config.json5 \
        -p examples/one_shot_schema_extraction/plugin_config.yaml

If you are using ``pixi``:

.. code-block:: shell

    pixi run compass process -c config.json5 \
        -p examples/one_shot_schema_extraction/plugin_config.yaml

Add ``-v`` (or ``-vv``) if you want log output in the terminal.
See the `Execution Basics example <https://natlabrockies.github.io/COMPASS/examples/execution_basics/README.html#running-a-compass-process>`_
for more details on running COMPASS pipelines.
