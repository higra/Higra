.. _arrays:

Arrays
======

.. important::

    ``#include "higra/structure/array.hpp``

The preferred way to represent homogeneous tabular data in higra is through the use of `xtensor <https://xtensor.readthedocs.io/en/latest/>`_ multi-dimentional arrays.
Xtensor aims at providing numpy like arrays in C++: take a look at the `numpy->xtensor cheat sheet <https://xtensor.readthedocs.io/en/latest/numpy.html>`_ .

Higra defines some commodity types (aliases) that map to particular xtensor types. There are 3 categories of types:

- points, ie. 1 dimensional arrays of fixed size: they do not require heap allocated memory and benefit from a lot of compile time optimization;
- fixed dimension arrays of variable size: they require heap allocated memory but, thanks to their fixed dimension, they also benefit from compile time optimization;
- variable dimension arrays of variable size: they require heap allocated memory.

Therefore, using the most specific type compatible with your needs will provide better performances.

.. list-table::
    :widths: 1 2 4 1
    :header-rows: 1

    *   - Typename
        - Template
        - Description
        - Base type
    *   - ``point``
        - ``value_t`` the type of elements, and ``dim`` the dimension of the space
        - An n dimensional point
        - ``xtensor_fixed``
    *   - ``point_2d_f``
        -
        - A two dimentional point with real coordinates (``double``)
        - ``xtensor_fixed``
    *   - ``point_2d_i``
        -
        - A two dimentional point with integral coordinates (``long``)
        - ``xtensor_fixed``
    *   - ``point_3d_f``
        -
        - A three dimentional point with real coordinates (``double``)
        - ``xtensor_fixed``
    *   - ``point_3d_i``
        -
        - A three dimentional point with real coordinates (``double``)
        - ``xtensor_fixed``
    *   - ``point_4d_f``
        -
        - A four dimentional point with real coordinates (``double``)
        - ``xtensor_fixed``
    *   - ``point_4d_i``
        -
        - A four dimentional point with real coordinates (``double``)
        - ``xtensor_fixed``
    *   - ``array1d``
        - ``value_t`` the type of elements
        - A one dimentional array
        - ``xtensor``
    *   - ``array2d``
        - ``value_t`` the type of elements
        - A two dimentional array
        - ``xtensor``
    *   - ``array3d``
        - ``value_t`` the type of elements
        - A three dimentional array
        - ``xtensor``
    *   - ``array4d``
        - ``value_t`` the type of elements
        - A four dimentional array
        - ``xtensor``
    *   - ``arraynd``
        - ``value_t`` the type of elements
        - A n-dimentional array (n being defined at runtime)
        - ``xarray``




