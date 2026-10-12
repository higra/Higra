Developing c++ extensions
=========================

While Higra provides many vectorized operators to implement algorithms efficiently in Python, it is possible that
some operations cannot be done efficiently in Python.

In such case, the `Higra-cppextension-cookiecutter <https://github.com/higra/Higra-cppextension-cookiecutter>`_ enables
to easily setup and generate c++ extension using Higra with Python bindings.

Binary compatibility with pip wheels
-----------------------------------

Compile extensions with C++17 and the headers exported by the installed Higra
wheel: :func:`~higra.get_include`, :func:`~higra.get_lib_include`, and
:func:`~higra.get_lib_cmake`. Use that environment's Python development headers.
Do not register Higra's ``Tree`` or ``UndirectedGraph`` classes again.

A minimal CMake project
-----------------------

This example selects pybind11 from the installed wheel explicitly. Run CMake
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

