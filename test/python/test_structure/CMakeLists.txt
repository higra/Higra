############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

set(PY_FILES
        __init__.py
        test_array_interop.py
        test_embedding.py
        test_lca_fast.py
        test_regular_graph.py
        test_tree.py
        test_undirected_graph.py)

REGISTER_PYTHON_MODULE_FILES("${PY_FILES}")

# Test-only extension module for the NumPy converters (higra/python/py_array.hpp).
# Never built for packaging: setup.py configures with HG_BUILD_WHEEL=ON and sets
# CMAKE_LIBRARY_OUTPUT_DIRECTORY to the wheel's extension directory.
if (NOT HG_BUILD_WHEEL)
    pybind11_add_module(_hg_array_test MODULE hg_array_test_module.cpp)
    target_include_directories(_hg_array_test PRIVATE
            ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib/include)
    # Explicit output directory (the generator expression prevents per-configuration
    # subdirectories), next to test_array_interop.py in the test build tree.
    set_target_properties(_hg_array_test PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY $<1:${CMAKE_CURRENT_BINARY_DIR}>
            RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_CURRENT_BINARY_DIR}>)
    foreach (config ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE})
        string(TOUPPER "${config}" config)
        set_target_properties(_hg_array_test PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY_${config} $<1:${CMAKE_CURRENT_BINARY_DIR}>
                RUNTIME_OUTPUT_DIRECTORY_${config} $<1:${CMAKE_CURRENT_BINARY_DIR}>)
    endforeach ()
    if (HG_USE_TBB)
        target_compile_definitions(_hg_array_test PRIVATE HG_USE_TBB)
        target_link_libraries(_hg_array_test PRIVATE TBB::tbb)
    endif ()
    # Built with the extension, so that the Python test suite always finds it.
    add_dependencies(higram _hg_array_test)
    # Tells test_array_interop.py that the module is expected in this directory (the source test
    # directory is also run against installed wheels, where the module does not exist).
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/_hg_array_test_expected "")
else ()
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/_hg_array_test_expected)
endif ()
