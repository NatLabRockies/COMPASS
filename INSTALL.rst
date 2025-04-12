Installing INFRA-COMPASS
========================

.. inclusion-install

Installing from PyPI
--------------------

INFRA-COMPASS can be installed via pip from
`PyPI <https://pypi.org/project/NREL-COMPASS>`__.

.. code-block:: shell

    pip install nrel-compass

.. note::

    You must have ``pip>=19.3`` to install from PyPI.

.. note::

    It is recommended to install and run INFRA-COMPASS from a virtual environment, for example,
    using `Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/main>`__.


Handling ImportErrors
---------------------

If you encounter an ``ImportError``, it usually means that Python couldn't find INFRA-COMPASS in the list of available
libraries. Python internally has a list of directories it searches through, to find packages. You can
obtain these directories with.

.. code-block:: python

    import sys
    sys.path

One way you could be encountering this error is if you have multiple Python installations on your system
and you don't have INFRA-COMPASS installed in the Python installation you're currently using.
In Linux/Mac you can run ``which python`` on your terminal and it will tell you which Python installation you're
using. If it's something like "/usr/bin/python", you're using the Python from the system, which is not recommended.
Instead, it is highly recommended to use an isolated environment, such as one created using
`Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/main>`__, for package and dependency updates.


Installing from Source
----------------------

If you would like to install and run INFRA-COMPASS from source, we recommend using
`pixi <https://pixi.sh/latest/>`_. This tool allows developers to install libraries and
applications in a reproducible way across multiple platforms. This means bugs are easier
to reproduce, and it's easier to move your development environment to a new piece of hardware.

We keep a version-controlled ``pixi.lock`` in the repository to allow locking with
the full requirements tree so we can reproduce behaviors and bugs and easily compare
results.

Start by cloning the ``COMPASS`` repository:

- Using ssh: :code:`git clone git@github.com:NREL/COMPASS.git`
- Using https: :code:`git clone https://github.com/NREL/COMPASS.git`

If you don't already have ``pixi`` installed on your system, follow the (simple) `installation
instructions <https://pixi.sh/latest/#installation>`_.

Once you have both ``pixi`` and the ``COMPASS`` source code, simply run:

.. code-block:: shell

    pixi shell

from the source code repository. You can now start using ``COMPASS``!

If you are planning to contribute to ``COMPASS``, you can use the ``pdev`` feature in ``pixi`` to
get all necessary python development tools:

.. code-block:: shell

    pixi shell -e pdev

To work on the Rust-based CLI, you can use the ``rdev`` feature instead:

.. code-block:: shell

    pixi shell -e rdev

You are welcome to use a different environment manager (e.g. ``conda``, ``mamba``, etc),
but we make no promises to provide support on environment-related issues/bugs in this case.


Preparing for Web Search
------------------------
Before you can perform any automated web search, you will need to install the browser
drivers that will be used to control this process. To do so, in your active conda environment
(or in your ``pixi shell``), run the following command:

.. code-block:: shell

    rebrowser_playwright install

This will download all required browser drivers. You only need to perform this step once
(i.e. during initial installation).
