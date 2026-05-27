"""Pytest config for the evals suite."""


def pytest_addoption(parser):
    """Add the ``--held-out`` flag to the test"""

    parser.addoption(
        "--held-out",
        action="store_true",
        default=False,
        help=("Run the eval against the held-out dataset"),
    )
