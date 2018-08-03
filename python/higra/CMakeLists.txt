############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

find_package(PythonInterp 2.7 REQUIRED)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/lib/" "${CMAKE_SOURCE_DIR}/lib/lib/cmake/pybind11/")

find_package(NumPy REQUIRED)
include("${CMAKE_SOURCE_DIR}/lib/lib/cmake/pybind11/pybind11Tools.cmake")

include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib/include ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIRS})

set(PYMODULE_COMPONENTS
        cpp/py_accumulators.cpp
        cpp/py_algo_tree.cpp
        cpp/py_embedding.cpp
        cpp/py_graph_image.cpp
        cpp/py_graph_weights.cpp
        cpp/py_hierarchy_core.cpp
        cpp/py_lca_fast.cpp
        cpp/py_log.cpp
        cpp/py_pink_io.cpp
        cpp/py_rag.cpp
        cpp/py_regular_graph.cpp
        cpp/py_tree_accumulators.cpp
        cpp/py_tree_graph.cpp
        cpp/py_tree_io.cpp
        cpp/py_undirected_graph.cpp
        cpp/py_watershed.cpp)

set(MODULE_FILES
        __init__.py
        data_cache.py
        graph_image.py
        graph_weights.py
        hierarchy_core.py
        pink_io.py
        rag.py
        tree_attributes.py
        tree_io.py
        watershed.py)

pybind11_add_module(higram pymodule.cpp ${PYMODULE_COMPONENTS})
set(UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} higram CACHE INTERNAL "" FORCE)


foreach (FILE ${MODULE_FILES})
    configure_file(${FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILE} COPYONLY)
endforeach ()