

# Higra: Hierarchical Graph Analysis

[![Build Status](https://travis-ci.org/PerretB/Higra.svg?branch=master)](https://travis-ci.org/PerretB/Higra) 
[![Build status](https://ci.appveyor.com/api/projects/status/5op4qm2cddm7iuj2/branch/master?svg=true)](https://ci.appveyor.com/project/PerretB/higra/branch/master)
[![codecov](https://codecov.io/gh/PerretB/Higra/branch/master/graph/badge.svg)](https://codecov.io/gh/PerretB/Higra)
[![Documentation Status](https://readthedocs.org/projects/higra/badge/?version=latest)](https://higra.readthedocs.io/en/latest/?badge=latest)

Higra is a C++/Python library for efficient graph analysis with a special focus on hierarchical methods. Some of the main features are:

- efficient methods and data structures to handle the dual representation of hierarchical clustering: dendrograms (trees) and ultra-metric distances (saliency maps);
- hierarchical clustering algorithms: agglomerative clustering (single-linkage, average-linkage, complete-linkage, or custom rule), hierarchical watersheds;
- various algorithms to manipulate and explore hierarchical clustering: accumulators, filtering/simplification, cluster extraction, (optimal) partitioning , alignment;
- algorithms on graphs: accumulators, computation of dissimilarities, partionning;
- assessment: supervised assessment of graph clustering and hierarchical clustering;
- image toolbox: special methods for grid graphs, hierarchical clustering methods dedicated to image analysis.

Higra is thought for modularity, performance and seamless integration with classical data analysis pipelines. The data structures (graphs and trees) are decoupled from data (vertex and edge weights ) which are simply arrays ([xtensor](https://github.com/QuantStack/xtensor) arrays in C++ and [numpy](https://github.com/numpy/numpy) arrays in Python).

## Installation

### Python frontend

The Python frontend can be installed with Pypi:

```bash
pip install higra
```

Supported systems: 

 - Python 3.5, 3.6, 3.7
 - Linux 64 bits, macOS, Windows 64 bits

### C++ backend

The C++ backend is an header only library. No facilities for system wide installation is currently provided: just copy/past where you need it!

## Demonstration and tutorials

Check the dedicated repository [Higra-Notebooks](https://github.com/PerretB/Higra-Notebooks) for a collection of Jupyter Notebooks dedicated to Higra.

## Build

### With cmake

Requires:

* cmake 
* Python + Numpy
* Boost Test (optional for unit testing of the C++ backend)
* Google Benchmark (optional for benchmarking of the C++ backend)

Commands:

```bash
git clone https://git.esiee.fr/perretb/Higra.git
mkdir build
cd build
cmake ../Higra/
make
```

Sometimes, cmake gets confused when several Python versions are installed on the system.
You can specify which version to use with `-DPYTHON_EXECUTABLE:FILEPATH=/PATH-TO-PYTHON/python`, e.g.

```
cmake -DPYTHON_EXECUTABLE:FILEPATH=/anaconda3/bin/python ../Higra/
```

The python package is build in the directory

```
build/python/
```

### With setup.py

The `setup.py` is a thin wrapper around the `cmake` script to provide compatibility with python `setuptools`. 

```
pip install cmake
python setup.py bdist_wheel
cd dist
pip install higra*.whl
```

