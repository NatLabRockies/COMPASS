"""Pytest config for the evals suite

Adds the ``--held-out`` flag so a single ``@pytest.mark.evals`` test
function can be pointed at either the dev or held-out dataset.

A dev run (default) compares its results against the committed
``results/dev/<name>_evals.json`` baseline and fails on aggregate or
per-row regression. A held-out run (``--held-out``) writes
``results/held_out/<name>_evals.json`` only and does **not** gate -- it's
an unbiased read taken before a release, not a CI tripwire.
"""


def pytest_addoption(parser):
    parser.addoption(
        "--held-out",
        action="store_true",
        default=False,
        help=(
            "Run the eval against the held-out dataset (no gate; "
            "release checkpoint only)."
        ),
    )
