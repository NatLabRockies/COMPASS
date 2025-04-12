.. _contributing:

Contributing to COMPASS
=======================

This document discusses working on the COMPASS code base and documentation.


Overview
--------

* All code changes should be submitted via a pull request (PR) and
  reviewed/approved by a core team member before merging  (see `Contributing Process`_ below for more details).
* Before setting up your environment, see the `Development environment guidelines`_.
* All code should adhere to `PEP8 <https://peps.python.org/pep-0008/>`_ (if you haven't
  read over it in a while, we recommend you skim it again for a refresher).
* All code should adhere to the `Stylistic guidelines`_.
* All code should have tests (see `Test coverage`_ below for more details).
* All code should be documented (see `Documentation`_ below for more details).


Contributing Process
--------------------

If you need a refresher on contributing code via GitHub using a pull request, check out the
`official GitHub documentation <https://docs.github.com/en/pull-requests>`_. There, you can
learn how to `create a branch <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository>`_,
`open a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_,
and `request a review <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review>`_.

If you are not sure where or how to start contributing to COMPASS, consider reaching out to the
current developer(s): **Paul Pinchuk** ["ppinchuk@nrel.gov"] or **Guilherme Pimenta Castelao** ["gpimenta@nrel.gov"].

When you are ready to contribute to COMPASS, clone a copy of the COMPASS repository from GitHub, check out your
own branch, and make the appropriate changes locally. Ensure that your new code adheres to all of the guidelines
below. When you are happy with your code, commit it locally. Ensure that you commit all tests that go along
with your code change.

Once all your code changes are committed locally and your code passes all tests, push it to the GitHub repository
and open a PR. Give your PR a short title and briefly describe your changes. Request that one of the core COMPASS
developers reviews your PR. We will likely ask you to make some modifications before you can merge. This is typical
and ensures that the quality of the overall codebase remains somewhat consistent.

Once your PR is approved, you may merge it into the main branch. If your code introduces significant new functionality
or fixes a critical bug, it may warrant a release. Please coordinate with a core COMPASS developer to create a new release,
which will automatically generate new wheels for installing COMPASS.


.. index-dev-link-end:

.. _dev-env-guidelines:
.. index-dev-link-start:


Development environment guidelines
----------------------------------

We use `pixi <https://pixi.sh/latest/>`_ to manage environments across developers.
This tool allows developers to install libraries and applications in a reproducible
way across multiple platforms. This means bugs are easier to reproduce, and it's easier
to move your development environment to a new piece of hardware.

We keep a version-controlled ``pixi.lock`` in the repository to allow locking with
the full requirements tree so we can reproduce behaviors and bugs and easily compare
results.

You can use the ``pdev`` feature in ``pixi`` to get all necessary python development tools::

    $ pixi shell -e pdev

To work on the Rust-based CLI, you can use the ``rdev`` feature instead::

    $ pixi shell -e rdev

You are welcome to use a different environment manager (e.g. ``conda``, ``mamba``, etc),
but we make no promises to provide support on environment-related issues/bugs in this case.


Stylistic guidelines
--------------------

We define a set of stylistic standards for COMPASS code development. The intent
is to maintain coherence when multiple developers contribute code to the repository.

Ruff
^^^^

Maintaining consistent code quality is crucial for COMPASS. To ensure uniformity and
adherence to coding standards, we employ the use `Ruff <https://docs.astral.sh/ruff/>`_.
Ruff is an "opinionated" formatter and linter designed to enhance code readability,
maintainability, and consistency that is extremely fast.

You can use the `Ruff VSCode extension <https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`_
if you are developing using VSCode. Alternatively, you can set a pre-commit hook to run Ruff.
This would perform automatic code formatting before any code is committed to the repository.
Both of these tools ensure that all code contributions meet the established quality standards,
minimizing the chances of introducing formatting inconsistencies or potential issues.


Imports
^^^^^^^

Use the following import conventions::

   import rex
   import numpy as np
   import pandas as pd


Test coverage
-------------

Pull requests (PRs) that modify code should either have new tests, or modify existing
tests to fail before the PR and pass afterwards.

You can run python COMPASS tests locally using pixi::

    $ pixi r -e ptest tests

Tests for a module should ideally cover all code in that module,
i.e., statement coverage should be at 100%, though this alone does not ensure that
your code is bug-free. Still, this is a good place to start, and you view the test
coverage at ``build/coverage`` by running::

    $ firefox build/coverage/index.html


Documentation
-------------

We strongly believe that documentation is a core part of code development, as it helps
both users of your function as well as other developers (including your future self).
As such, please adhere to these guidelines:

1) Document all public functions and classes
    Public functions and classes are defined as not having any leading underscores (``_``).
    These functions are detected by Sphinx and therefore should have docstrings formatted according to the
    `NumPy documentation style <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
2) Do not include a period (``.``) on the first line (short summary) of a docstring.
    This is a stylistic decision particular to the COMPASS codebase.
3) Do not include a short summary (first line docstring) for ``__init__`` methods.
    Instead, document any object summaries using the class docstring. You can and should still document
    initialization parameters in the ``__init__`` docstring. See any the docstring of the main COMPASS objects
    (e.g. :class:`~compass.services.openai.OpenAIService`) for an example.
4) Protected/private functions should contain minimal documentation.
    Public functions and classes are defined as having one or more leading underscores (``_``).
    These functions are **not** detected by Sphinx and therefor should contain minimal documentation
    (typically a docstring with just a single sentence). Do not include *any* sections from the
    NumPy documentation style. With minimal exceptions,
    we treat protected and private functions as implementation details. As such, if you did not
    write the function, you should probably not be modifying/calling/touching it in any way.
    Such code is subject to change at any time, so you should never rely on private/protected
    functionality unless you know what you are doing (in which case you should be relying on the
    function's code, not docstring).
5) Link any functions and/or classes that you reference in your docstring.
    Sphinx allows interlinks between different sets of documentation, which can be a really convenient
    way for new users to learn more about the external libraries they are expected to use. For more
    information on how to set up links in your documentation, please see
    `this short blog post <https://kevin.burke.dev/kevin/sphinx-interlinks/>`_. In particular,
    we use the ``:func:`` directive for standalone functions, ``:meth:`` for class methods,
    ``:class:`` for references to classes, and ``:obj:`` for all other links. Please use this
    list of available COMPASS intersphinx mappings:

        * COMPASS: ``compass``
            For example, use ``:func:`~compass.scripts.process.process_counties_with_openai```,
            which renders as :func:`~compass.scripts.process.process_counties_with_openai`
        * Pandas: ``pandas``
            For example, use ``:obj:`~numpy.array```, which renders as :obj:`~numpy.array`
        * MatplotLib: ``matplotlib``
            For example, use ``:func:`~matplotlib.pyplot.plot```, which renders as :func:`~matplotlib.pyplot.plot`
        * Plotly: ``plotly``
            For example, use ``:class:`plotly.graph_objects.Figure```, which renders as :class:`plotly.graph_objects.Figure`
        * Networkx: ``networkx``
            For example, use ``:class:`~networkx.MultiDiGraph```, which renders as :class:`~networkx.MultiDiGraph`
        * elm: ``elm``
            For example, use ``:class:`elm.web.document.PDFDocument```, which renders as :class:`elm.web.document.PDFDocument`


To check your docstring additions/updates, you can build a local version of the HTML documentation::

    $ pixi r -e pdoc make-html

After running this command, simply open ``docs/_build/html/index.html`` using your favorite browser, e.g.::

    $ firefox docs/_build/html/index.html


Miscellaneous
-------------

A collection of other miscellaneous guidelines.


Updating ``pyproject.toml``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because we statically link the requirements, the compilation process can
extend to 30-60 minutes. That is mostly due to ``duckdb`` and ``tokio``. To
optimize this process, we use a GiHub Actions cache, but too many images blow
the quota make the process less useful. To avoid this, the Rust component is setup
to only save the compiled cache for branch main, and all other branches initiate
from that, and hopefully save some fair amount of time. Thus, a good practice here
is to update the `pyproject.toml` file and/or any Rust dependencies in a dedicated
branch to avoid too much recompilation in working branches.


Error Handling
^^^^^^^^^^^^^^

Do not throw default warning/errors. Always use some subclass of
``compass.warn.COMPASSWarning`` or ``compass.exceptions.COMPASSError``, like so::

    from warnings import warn

    from compass.warn import COMPASSWarning
    from compass.exceptions import COMPASSValueError

    ...

    def my_func():
        if not_good_enough_data:
            warn("Watch out for your data!", COMPASSWarning)

        if unacceptable_value:
            raise COMPASSValueError("This value is unacceptable")


This allows maximum flexibility for downstream users of the library.
In particular, they can choose whether to handle just errors that originate
from COMPASS (i.e. by catching ``COMPASSValueError``) or to handle the generic
version of the error (i.e. by catching ``ValueError``).

In addition, all COMPASS warnings and errors have a logging call built in.
This means you no longer need to do something like this::

    # BAD - do not do this
    if unacceptable_value:
        msg = "This value is unacceptable"
        logger.exception(msg)
        raise ValueError(msg)

Simply raising ``COMPASSValueError`` with the appropriate message performs the
logging call shown above, internally, every time.


Test File Structure
^^^^^^^^^^^^^^^^^^^

All test files (e.g. ``test_scenario.py``) should start/end with the following block of code::

    from pathlib import Path
    import pytest

    ...

    if __name__ == "__main__":
        pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])


This allows the (single) file to be executed, running only the tests contained
within. This is extremely useful when updating/modifying/adding tests in the file.
