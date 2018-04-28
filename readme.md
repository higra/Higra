# Higra: Hierarchical Graph Analysis

## Build

Requires:

* cmake 
* BOOST
* Python + Numpy

Commands:

```bash
git clone https://git.esiee.fr/perretb/Higra.git
mkdir build
cd build
cmake ../Higra/
make
```

Sometimes, cmake gets confused when several Python versions are installed in the system.
You can specify which version to use with `-DPYTHON_EXECUTABLE:FILEPATH=/PATH-TO-PYTHON/python`, e.g.

```
cmake -DPYTHON_EXECUTABLE:FILEPATH=/anaconda3/bin/python ../Higra/
```

The python package is build in the directory

```
build/python/
```
