.. _arrays:

Arrays
======

.. important::

    ``#include "higra/structure/array.hpp"``

In C++, Higra stores numerical data, such as vertex weights, edge weights or node attributes, in ``hg::array``:
a multidimensional array of elements of the same type. Its memory layout is the same as a C-contiguous (row-major)
NumPy array, so results move from C++ to Python without copying, and C-contiguous NumPy arrays are read by C++
without copying (see :ref:`extensions`).

Types
-----

.. list-table::
    :widths: 2 5
    :header-rows: 1

    *   - Type
        - Description
    *   - ``array<T>``
        - Owning array of elements of type ``T``. Copying an array copies its elements; moving an array transfers them.
    *   - ``array_view<const T>``
        - Read-only view of the elements of an array (or of any contiguous buffer). A view never owns its elements.
    *   - ``array_view<T>``
        - Mutable view: like ``array_view<const T>``, but the elements can be modified through it.
    *   - ``array_1d<T>``, ``array_2d<T>``, ``array_3d<T>``, ``array_4d<T>``, ``array_nd<T>``
        - Other names of ``array<T>``. They document the number of dimensions expected by a function, but they are
          all the same type: the number of dimensions is a run-time property of every array.
    *   - ``point<T, dim>``, ``point_2d_i``, ``point_2d_f``, ... ``point_4d_f``
        - Fixed-size point of ``dim`` coordinates stored inline (no allocation), with elementwise ``+``, ``-``,
          ``min`` and ``max`` (``_i``: ``index_t`` coordinates, ``_f``: ``double`` coordinates).

Every array has between 0 and 8 dimensions. Its elements are stored contiguously in row-major order, like a NumPy
array created with ``order="C"``. The element type ``T`` can be ``bool``, any integer type, ``float`` or ``double``.
Shapes, sizes and indices are signed integers (``index_t``, a 64-bit integer).

Creating arrays
---------------

*From values:*

.. code-block:: cpp
    :linenos:

    using namespace hg;

    array<int> a1{1, 2, 3};          // 1d array of 3 elements
    array<int> a2{{1, 2, 3},
                  {4, 5, 6}};        // 2d array of shape (2, 3)

Braces always give the **elements** of the array, never its shape: ``array<int>{5}`` is the 1d array ``[5]``.
Literals have up to 4 dimensions; rows of different lengths throw an exception.

*From a shape:*

.. code-block:: cpp
    :linenos:

    auto b1 = empty<double>({2, 3});     // shape (2, 3), elements not initialized
    auto b2 = zeros<double>({2, 3});     // filled with 0
    auto b3 = ones<index_t>({10});       // filled with 1
    auto b4 = full<float>({4, 4}, 0.5f); // filled with 0.5
    auto b5 = arange<index_t>(5);        // {0, 1, 2, 3, 4}
    auto b6 = arange<double>(0, 1, 0.25);// {0, 0.25, 0.5, 0.75}

    array<double> b7(array_shape(2, 3), 5.0); // same as full<double>({2, 3}, 5)

The functions ``empty_like``, ``zeros_like``, ``ones_like`` and ``full_like`` create an array with the shape and
element type of another array.

.. attention::

    ``array<double> b(array_shape(n));`` declares a function, not an array (C++'s "most vexing parse").
    Write ``auto b = empty<double>({n});`` or ``array<double> b{array_shape(n)};`` instead.

Properties and element access
-----------------------------

.. code-block:: cpp
    :linenos:

    array<int> a{{1, 2, 3},
                 {4, 5, 6}};

    a.dimension();   // 2
    a.size();        // 6, the number of elements
    a.shape();       // (2, 3)
    a.shape(1);      // 3
    a.empty();       // false
    a.row_size();    // 3, the number of elements per row: size() / shape(0)

    a(1, 2) = 7;     // element at row 1, column 2: one index per dimension
    int v = a[5];    // element at position 5 in row-major order (here a(1, 2))

    int *p = a.data();       // pointer to the first element
    for (auto &x : a) {      // all elements in row-major order
        x += 1;
    }

Element access is not checked by default. Defining the macro ``HG_ARRAY_CHECKS`` enables rank and bounds checks on
``operator()``, ``operator[]`` and ``row``. Higra's CMake project defines it in every build configuration except
Release (for example Debug and RelWithDebInfo).

Arrays can be printed with ``std::cout << a``; large arrays are summarized.

Views
-----

A view gives access to the elements of an array without copying them. It stores a pointer to the elements and a
shape, so it is cheap to create and to pass by value. A view is only valid while the elements it refers to exist:
it must not outlive its array, and it must not be used after the array is resized, assigned or moved.

.. code-block:: cpp
    :linenos:

    array<double> a = zeros<double>({4, 3});

    array_view<const double> v = a;   // read-only view of all elements
    auto r = a.row(1);                // the 3 elements a(1, 0), a(1, 1), a(1, 2): shape (3)
    auto s = a.slice(1, 3);           // rows 1 and 2: shape (2, 3)
    auto f = a.flat();                // all elements as a 1d view: shape (12)
    auto m = a.reshaped({3, 4});      // same elements with shape (3, 4)

    s.fill(1.0);                      // sets rows 1 and 2 of a to 1
    array<double> copy(s);            // owning copy of the elements of a view

A view of elements stored elsewhere, for example in a buffer owned by another library, is created from a pointer and
a shape: ``array_view<const double> v(ptr, {4, 3});``. The pointer must refer to at least as many elements as the
product of the dimensions, and this product must fit in an ``index_t``. Neither condition is checked.

A view obtained from a mutable array is mutable (``array_view<double>`` above); a view obtained from a const array is
read-only. Creating a mutable view of a const array or of a temporary array is a compile-time error. A read-only view
of a temporary array is allowed, so that temporaries can be passed to functions, but it becomes invalid at the end of
the statement: do not store it.

``row``, ``slice``, ``reshaped`` and ``flat`` cover contiguous sub-arrays only. To access other parts of an array,
such as a column, write a loop.

Functions
---------

The functions below are in namespace ``hg``. They take ``array_view`` arguments, so they accept arrays and views
alike, and compute their result immediately. Array arithmetic is written with ``map`` or with an explicit loop:
there are no arithmetic operators on arrays.

.. list-table::
    :widths: 2 5
    :header-rows: 1

    *   - Function
        - Description
    *   - ``assign(dst, src)``
        - Copies the elements of ``src`` into the mutable view ``dst`` (same shape), converting them to the element
          type of ``dst``.
    *   - ``cast<U>(a)``
        - New array with the elements of ``a`` converted to ``U``.
    *   - ``map(f, a, b, ...)``
        - New array ``r`` with ``r[i] = f(a[i], b[i], ...)``; all inputs have the same shape.
    *   - ``map_into(out, f, a, b, ...)``
        - Same as ``map``, writing into the existing mutable view ``out``. ``out`` may be the same view as an input
          (in-place update), but must not overlap an input in any other way.
    *   - ``min(a)``, ``max(a)``, ``minmax(a)``
        - Smallest and largest elements; throw on an empty array.
    *   - ``argmin(a)``, ``argmax(a)``
        - Position (in row-major order) of the first smallest or largest element.
    *   - ``sum(a)``, ``sum<Acc>(a)``
        - Sum of the elements. Integers are summed in 64-bit integers, floating point values in their own type,
          unless an accumulator type ``Acc`` is given.
    *   - ``count_nonzero(a)``, ``all(a)``, ``any(a)``
        - Number of non-zero elements; whether all or any elements are non-zero.
    *   - ``take(a, idx)``
        - New array with the rows ``a[idx[0]], a[idx[1]], ...`` (NumPy's ``a[idx]``).
    *   - ``put(a, idx, v)``, ``put(a, idx, values)``
        - Sets the rows ``a[idx[i]]`` to the value ``v``, or to the rows of ``values`` (NumPy's ``a[idx] = ...``).
          ``idx``, ``v`` and ``values`` are read before ``a`` is modified, so they may refer to elements of ``a``.
          With repeated indices, the last write wins.
    *   - ``a == b``, ``a != b``
        - Whether two arrays have the same shape and equal elements.
    *   - ``allclose(a, b, rtol, atol)``
        - Whether two arrays have the same shape and elements equal within a tolerance (as ``numpy.allclose``).
    *   - ``same_shape(a, b)``
        - Whether two arrays have the same shape.

.. code-block:: cpp
    :linenos:

    array<double> x{1, 2, 3};
    array<double> y{4, 5, 6};

    auto z = map([](double a, double b) { return a * b + 1; }, x, y); // {5, 11, 19}
    auto total = sum(z);                                               // 35
    auto rows = take(z, array<index_t>{2, 0});                         // {19, 5}

Parallel execution
------------------

When Higra is compiled with Intel TBB (``HG_USE_TBB``), copies, conversions, ``take``, reductions and comparisons of
large arrays (at least 65,536 elements) run on several threads. Filling an array runs on the calling thread.
``put`` writes the elements on the calling thread, in the order of the indices, but may check the indices, and copy
inputs that refer to elements of the destination, on several threads.

``map`` and ``map_into`` call their function on the calling thread, one element after the other, because the
function may use objects that are not safe to use from several threads. When the function can safely be called
concurrently, request parallel execution explicitly:

.. code-block:: cpp
    :linenos:

    auto r = map(exec::par, [](double v) { return std::sqrt(v); }, x);

Results do not depend on parallel execution: all functions give identical results, with or without TBB and with any
number of threads. This includes floating point sums, which always add the elements in the same order.
In Python, the number of threads is set with :func:`~higra.set_num_threads`.

Writing algorithms
------------------

Functions that read arrays take an ``array_view<const T>`` by value. Such a parameter accepts arrays, views,
const arrays and temporaries, with a single instantiation of the function per element type. Functions that modify
an array in place take an ``array_view<T>``. Results are returned as ``array`` by value, which never copies the
elements.

.. code-block:: cpp
    :linenos:

    #include "higra/graph.hpp"
    using namespace hg;

    // read-only input, new result
    template<typename T>
    array<double> normalize(array_view<const T> weights) {
        const auto mm = minmax(weights);
        const T lo = mm.first;
        const T hi = mm.second;
        double range = (hi > lo) ? double(hi - lo) : 1.0;
        return map([lo, range](T v) { return (v - lo) / range; }, weights);
    }

    // in-place modification
    template<typename T>
    void clip(array_view<T> weights, T lo, T hi) {
        for (auto &v : weights) {
            v = std::min(std::max(v, lo), hi);
        }
    }

    array<float> w{3, 1, 4, 1, 5};
    auto n = normalize(w);                 // {0.5, 0, 0.75, 0, 1}
    clip(w, 2.f, 4.f);                     // w = {3, 2, 4, 2, 4}
    auto n2 = normalize(w.slice(0, 3));    // views are accepted as well

Data associated with a graph or a tree is stored with one row per vertex, edge or node: the first dimension of the
array has the number of vertices, edges or nodes, and the other dimensions, if any, hold the features of each
element. The features of element ``i`` are ``a.row(i)``, or the ``a.row_size()`` elements starting at
``a.data() + i * a.row_size()``.

Never return a view of a local array: the array is destroyed when the function returns. Return the array itself.
