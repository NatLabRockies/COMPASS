Installing INFRA-COMPASS
========================

.. inclusion-install

Installing from PyPI
--------------------

INFRA-COMPASS can be installed via pip from
`PyPI <https://pypi.org/project/INFRA-COMPASS>`__.

.. code-block:: shell

    pip install infra-compass

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

- Using ssh: :code:`git clone git@github.com:NatLabRockies/COMPASS.git`
- Using https: :code:`git clone https://github.com/NatLabRockies/COMPASS.git`

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

    playwright install
    rebrowser_playwright install
    patchright install

This will download all required browser drivers. You only need to perform this step once
(i.e. during initial installation). If you see any warning messages about missing libraries,
you may have to run the following command (requires ``sudo``):

.. code-block:: shell

    playwright install-deps


Preparing for Docling-RS PDF Parsing
------------------------------------
Docling-RS is an optional backend for native document conversion. Its
dependencies are installed by default on Linux and Windows, both through
pip and in the standard Python Pixi environments. On macOS, these
dependencies are excluded automatically. See `Intel macOS: ONNX Runtime Setup`_
if you want to install Docling-RS manually on macOS.

To select this backend on a platform where Docling is installed, set
``COMPASS_FILE_LOAD_BACKEND=docling`` before starting COMPASS. Before parsing
PDFs or scanned images, download the model assets once from the environment
where Docling is installed:

.. code-block:: shell

    python -c "from docling_rs import download_models; download_models()"

If you are using Pixi, prefix this command with ``pixi run``.

Docling models are cached under ``~/.cache/docling.rs`` by default. Set
``DOCLING_RS_CACHE_DIR`` to use a different cache directory, and make that
cache available to the worker processes. HTML and Markdown conversion do
not require these models.

Some scanned PDFs use image encodings that the native PDF renderer cannot
read completely. If an OCR-enabled PDF converts successfully but produces
no Markdown, COMPASS renders its pages with Poppler and retries them with
the same Docling-RS OCR engine. This retry preserves the original PDF and
page count, respects explicit OCR opt-outs, and remains subject to the
subprocess deadline. Poppler is included in the Pixi environment.


Intel macOS: ONNX Runtime Setup
-------------------------------

COMPASS does not install Docling on macOS. If you explicitly install
``docling-rs`` and ``docling-core``, Docling-RS is built from source.
Its Rust ONNX Runtime dependency does not provide prebuilt binaries for
``x86_64-apple-darwin``, so Intel Macs must provide a shared
ONNX Runtime library before installing COMPASS.

Install ONNX Runtime with `Homebrew <https://brew.sh/>`_ and configure the
build in the terminal where you will install and run COMPASS:

.. code-block:: shell

    brew install onnxruntime
    export ORT_LIB_LOCATION="$(brew --prefix onnxruntime)/lib"
    export ORT_PREFER_DYNAMIC_LINK=1
    export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Wl,-rpath,$ORT_LIB_LOCATION"

``ORT_LIB_LOCATION`` identifies the directory containing
``libonnxruntime.dylib``. ``ORT_PREFER_DYNAMIC_LINK`` selects that shared
library instead of downloading a prebuilt Rust dependency. The ``RUSTFLAGS``
setting embeds its location in the native extension's runtime search path,
so Python can find it when importing Docling-RS.

Homebrew may build ONNX Runtime from source if an Intel macOS bottle is
unavailable. Building Docling-RS also requires a Rust toolchain compatible
with the installed release. Keep the Homebrew library installed while
using COMPASS. Set these variables again in each new terminal before
installing or rebuilding dependencies, or add them to your shell profile.


After this setup, install the backend explicitly in the Python environment
used to run COMPASS:

.. code-block:: shell

    python -m pip install "docling-rs>=1.104.3,<2" "docling-core>=2.101.1,<3"


For a Pixi development environment, use this command instead:

.. code-block:: shell

    pixi run -e pdev --locked uv pip install --python python "docling-rs>=1.104.3,<2" "docling-core>=2.101.1,<3"

Replace ``pdev`` with the environment you use to run COMPASS, and use that
same environment when downloading models.
