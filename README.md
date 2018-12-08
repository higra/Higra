[![Build Status](https://travis-ci.org/PerretB/Higra.svg?branch=master)](https://travis-ci.org/PerretB/Higra)

# Higra: Hierarchical Graph Analysis

## Build

Requires:

* cmake 
* Python + Numpy
* Boost Test (optional)
* Google Benchmark (optional)

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
