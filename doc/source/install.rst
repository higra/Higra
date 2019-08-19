Installation
============

Pre-build binaries
------------------

The Python package can be installed with Pypi:

.. code-block:: bash

    pip install higra


Supported systems:

 - Python 3.4, 3.5, 3.6, 3.7
 - Linux 64 bits, macOS, Windows 64 bits

Manual build
------------

Higra can be build from source directly with `cmake <https://cmake.org/>`_ or through
a wrapper `setuptools <https://setuptools.readthedocs.io/en/latest/>`_
``setup.py`` script. The latter is especially useful to create wheels (pip installable packages).

Building Higra from source requires:

    * a c++ 14 compiler (tested with GCC, Clang, MSVC)
    * cmake (2.8+)
    * Python (3.4+) with Numpy (1.15.4+)

With cmake
**********

The following commands will download and build Higra from source.
The python package will be in the directory ``build/higra/``.
Note that the python package must be *findable* by Python in order to be used
(e.g. by setting your `PYTHONPATH <https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH>`_
environment variable).

.. code-block:: bash

    git clone https://github.com/higra/Higra.git
    mkdir build
    cd build
    cmake  -DDO_AUTO_TEST=ON ../Higra/
    make


Sometimes, cmake gets confused when several Python versions are installed on the system.
You can specify which version to use with ``-DPYTHON_EXECUTABLE:FILEPATH=/PATH-TO-PYTHON/python``, e.g.

.. code-block:: bash

    cmake -DDO_AUTO_TEST=ON -DPYTHON_EXECUTABLE:FILEPATH=/anaconda3/bin/python ../Higra/


Cmake options:
++++++++++++++

- ``USE_SIMD`` (boolean, default ``ON``): Use SIMD instructions
- ``DO_CPP_TEST`` (boolean, default ``ON``): Build the c++ test suit
- ``DO_AUTO_TEST`` (boolean, default ``OFF``): Execute test suit automatically at the end of the build

With setuptools
***************

The file ``setup.py`` is a thin wrapper around the cmake script.
The following commands will download the library, create a binary wheel and install it with *pip*.


.. code-block:: bash

    git clone https://github.com/higra/Higra.git
    cd Higra
    python setup.py bdist_wheel
    cd dist
    pip install higra*.whl


