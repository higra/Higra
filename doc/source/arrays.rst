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
        - A three dimentional point with integral coordinates (``long``)
        - ``xtensor_fixed``
    *   - ``point_4d_f``
        -
        - A four dimentional point with real coordinates (``double``)
        - ``xtensor_fixed``
    *   - ``point_4d_i``
        -
        - A four dimentional point with integral coordinates (``long``)
        - ``xtensor_fixed``
    *   - ``array_1d``
        - ``value_t`` the type of elements
        - A one dimentional array
        - ``xtensor``
    *   - ``array_2d``
        - ``value_t`` the type of elements
        - A two dimentional array
        - ``xtensor``
    *   - ``array_3d``
        - ``value_t`` the type of elements
        - A three dimentional array
        - ``xtensor``
    *   - ``array_4d``
        - ``value_t`` the type of elements
        - A four dimentional array
        - ``xtensor``
    *   - ``array_nd``
        - ``value_t`` the type of elements
        - A n-dimentional array (n being defined at runtime)
        - ``xarray``



Quick start
-----------

Creating arrays
***************

*From scratch:*

.. code-block:: cpp
    :linenos:

    // explicit initialization
    array_2d<int> a1 {{1, 2, 3},
                      {4, 5, 6}};

    // empty (non initialized) array of given shape
    array_2d<int> a2 = array_1d<int>::from_shape({2, 3});

    // array of given shapes initialized with given value
    array_2d<int> a3({2, 3}, 5);

    // array of given shapes initialized with 0
    array_2d<int> a4 = xt::zeros<int>({2, 3});

    // array of given shapes initialized with 1
    array_2d<int> a5 = xt::ones<int>({2, 3});

*From an existing array:*

.. code-block:: cpp
    :linenos:


    array_2d<int> a1 {{1, 2, 3},
                      {4, 5, 6}};

    // same shape and value type, non-initialized
    auto a2 = xt::empty_like(a1);

    // same shape and value type, initialized to 0
    auto a3 = xt::zeros_like(a1);

    // same shape and value type, initialized to 1
    auto a4 = xt::ones_like(a1);

    // same shape and value type, initialized to given value
    auto a5 = xt::full_like(a1, 5);

Properties
**********

.. code-block:: cpp
    :linenos:


    array_2d<int> a1 {{1, 2, 3},
                      {4, 5, 6}};

    // dimension
    auto d = a1.dimension(); // 2

    // size
    auto s = a1.size(); // 6

    // shape
    auto sh = a1.shape();
    sh[0]; // 2
    sh[1]; // 3

Element access
**************

.. code-block:: cpp
    :linenos:


    array_2d<int> a1 {{1, 2, 3},
                      {4, 5, 6}};

    // modify and read element at line 1, column 2
    a1(1, 2) = 7;
    int v = a1(1, 2);

    // Same thing with the [] operator
    a1[{1, 2}] = 7;
    int v = a1[{1, 2}];

Display
*******

.. code-block:: cpp
    :linenos:

    #include <iostream>
    #include <xtensor/xio.hpp>
    using namespace std;

    array_2d<int> a1 {{1, 2, 3},
                      {4, 5, 6}};

    cout << a1 << "\n";

Lazy Evaluation
***************

Most xtensor operations are lazily evaluated. In the following situation:

.. code-block:: cpp
    :linenos:


    array_1d<int> a1{1, 2, 3};
    array_1d<int> a2{4, 5, 6};

    auto r = a2 * (a1 + a2);

`r` does only store the expression, i.e. the symbolic operation, but no actual results. The result will be computed when elements are accessed:


.. code-block:: cpp
    :linenos:

    auto v = r(1); // performs 5 * (2 + 5)

If an expression is evaluated several times at the same position, the same result will be computed several times.

An expression can be fully evaluated by assigning it to an actual array or by using the `eval` function:

.. code-block:: cpp
    :linenos:

    array_1d<int> ar1 = r;  // evaluates the expression r and assigns the result to ar1
    auto ar2 = xt::eval(r); // evaluates r and assigns it to an array called ar2

.. attention::

    Returning a non evaluated expression that refers to local variable will lead to undefined behaviors:

    .. code-block:: cpp
        :linenos:

        auto function(){
            array_1d<int> a1{1, 2, 3};
            array_1d<int> a2{4, 5, 6};
            //return a1 + a2;  => ERROR depends of local variables that are destructed at the end of the function
            return xt::eval(a1 + a2); // OK
        }





