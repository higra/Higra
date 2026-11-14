.. _extensions:

Developing c++ extensions
=========================

While Higra provides many vectorized operators to implement algorithms efficiently in Python, it is possible that
some operations cannot be done efficiently in Python.

In such case, the `Higra-cppextension-cookiecutter <https://github.com/higra/Higra-cppextension-cookiecutter>`_ enables
to easily setup and generate c++ extension using Higra with Python bindings.

Binary compatibility with pip wheels
------------------------------------

Compile extensions with C++17 and the headers exported by the installed Higra
wheel: :func:`~higra.get_include`, :func:`~higra.get_lib_include`, and
:func:`~higra.get_lib_cmake`. Use that environment's Python development headers.
Do not register Higra's ``Tree`` or ``UndirectedGraph`` classes again.

Exchanging arrays with Python
-----------------------------

Higra's C++ arrays (see :ref:`arrays`) are converted to and from NumPy arrays by the header
``higra/python/py_array.hpp``. Include it in every source file that binds a function taking or returning
``hg::array`` or ``hg::array_view``. NumPy itself is only needed at run time: the extension does not compile against
NumPy's headers.

A minimal extension, ``extension.cpp``:

.. code-block:: cpp
    :linenos:

    #include <algorithm>
    #include <pybind11/pybind11.h>
    #include "higra/graph.hpp"
    #include "higra/python/py_array.hpp"

    namespace py = pybind11;
    using namespace hg;

    // read-only input, new result
    template<typename T>
    array<double> normalize(array_view<const T> weights) {
        auto [lo, hi] = minmax(weights);
        double range = (hi > lo) ? double(hi - lo) : 1.0;
        return map([lo, range](T v) { return (v - lo) / range; }, weights);
    }

    // in-place modification
    void clip(array_view<double> weights, double lo, double hi) {
        for (auto &v : weights) {
            v = std::min(std::max(v, lo), hi);
        }
    }

    // Higra objects as input
    array<index_t> depth(const tree &t) {
        auto d = zeros<index_t>({num_vertices(t)});
        for (auto n : root_to_leaves_iterator(t, leaves_it::include, root_it::exclude)) {
            d(n) = d(parent(n, t)) + 1;
        }
        return d;
    }

    PYBIND11_MODULE(my_higra_extension, m) {
        // one overload per supported element type
        m.def("normalize", &normalize<float>, py::arg("weights"));
        m.def("normalize", &normalize<double>, py::arg("weights"));
        m.def("clip", &clip, py::arg("weights"), py::arg("lo"), py::arg("hi"));
        m.def("depth", &depth, py::arg("tree"));
    }

Its use from Python:

.. code-block:: python
    :linenos:

    import numpy as np
    import higra as hg
    import my_higra_extension as ext

    w = np.array([3, 1, 4, 1, 5], dtype=np.float64)
    ext.normalize(w)        # new array: [0.5, 0, 0.75, 0, 1]
    ext.normalize(w[::2])   # any memory layout is accepted: [0, 0.5, 1]
    ext.clip(w, 2, 4)       # w is modified: [3, 2, 4, 2, 4]

    tree = hg.Tree(np.array([5, 5, 6, 6, 6, 7, 7, 7]))
    ext.depth(tree)         # [2, 2, 2, 2, 2, 1, 1, 0]

Function parameters
*******************

.. list-table::
    :widths: 2 5
    :header-rows: 1

    *   - C++ parameter
        - Accepted Python values
    *   - ``array_view<const T>``
        - A NumPy array whose dtype matches ``T``, with any memory layout (strided, reversed, transposed...);
          when no overload matches the dtype exactly, anything NumPy can convert to an array of ``T``
          (lists, scalars, arrays of another dtype). The view is valid only during the call.
    *   - ``array<T>``
        - The same values; the elements are copied into an array owned by C++, which can be kept after the call.
    *   - ``array_view<T>`` (in-place)
        - Only a writeable, C-contiguous NumPy array whose dtype matches ``T`` exactly. Modifications are visible in
          Python. Any other value raises ``TypeError``.

Element types map to NumPy dtypes of the same kind and size: ``bool``, signed and unsigned integers of 8 to 64 bits,
``float`` (``float32``) and ``double`` (``float64``). Arrays with more than 8 dimensions raise ``ValueError``.

When a function is bound once per element type, as ``normalize`` above, a NumPy array calls the overload of its
own dtype, whatever its memory layout. Values that match no overload exactly, such as lists or arrays of other
dtypes, are converted for the first registered overload that accepts them: here ``ext.normalize([1, 2, 3])`` calls
``normalize<float>``.

An in-place function only accepts arrays that it can modify directly. To support arrays with another memory layout
or dtype, call it on a contiguous copy in a Python wrapper and copy the result back:

.. code-block:: python
    :linenos:

    def clip(weights, lo, hi):
        tmp = np.array(weights, dtype=np.float64, order="C")
        ext.clip(tmp, lo, hi)
        weights[...] = tmp

Return values
*************

.. list-table::
    :widths: 2 5
    :header-rows: 1

    *   - C++ return value
        - Python result
    *   - ``array<T>`` (by value)
        - A new writeable NumPy array. The elements are not copied: the NumPy array takes ownership of them.
    *   - ``const array<T>&`` or ``array_view<const T>``, with ``py::return_value_policy::reference_internal``
        - A read-only NumPy array that shares the memory of the C++ object; the Python object that owns it (``self``
          for a method) is kept alive as long as the NumPy array exists.
    *   - ``array<T>&`` or ``array_view<T>``, with ``py::return_value_policy::reference_internal``
        - The same, but writeable.
    *   - A reference or a view, with any other return value policy
        - A new NumPy array holding a copy of the elements.
    *   - ``std::pair``, ``std::tuple`` or ``std::vector`` of arrays
        - A tuple or list of NumPy arrays, following the rules above (``std::vector`` requires
          ``#include <pybind11/stl.h>``).

Return a view only with ``reference_internal`` and only when it refers to data owned by the bound object:
a view of a local array becomes invalid when the function returns.

A minimal CMake project
-----------------------

This example builds the extension above. It selects pybind11 from the installed wheel explicitly. Run CMake
outside the Higra source checkout and select the Python environment containing
your pip installation with ``-DPython_EXECUTABLE=/path/to/python``.

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.18)
    project(my_higra_extension LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import higra as hg; print(hg.get_include()); print(hg.get_lib_include()); print(hg.get_lib_cmake())"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        RESULT_VARIABLE hg_status OUTPUT_VARIABLE hg_paths
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT hg_status EQUAL 0)
        message(FATAL_ERROR "Cannot import installed Higra")
    endif()
    string(REPLACE "\n" ";" hg_paths "${hg_paths}")
    list(GET hg_paths 0 hg_include)
    list(GET hg_paths 1 hg_lib_include)
    list(GET hg_paths 2 hg_cmake)
    find_package(pybind11 3.1.0 EXACT CONFIG REQUIRED
        PATHS "${hg_cmake}/pybind11" NO_DEFAULT_PATH)
    pybind11_add_module(my_higra_extension extension.cpp)
    target_include_directories(my_higra_extension BEFORE PRIVATE
        "${hg_include}" "${hg_lib_include}")

