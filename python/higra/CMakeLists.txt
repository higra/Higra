

find_package(Boost REQUIRED COMPONENTS unit_test_framework)
find_package(PythonInterp 3.5 REQUIRED)


include_directories(${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/lib/include ${PYTHON_INCLUDE_DIR})
# ${PROJECT_SOURCE_DIR}/lib/pybind11/include
#pybind11_add_module(xtensor_python_test example_xtensor.cpp)

set(PYMODULE_COMPONENTS
        py_undirected_graph.cpp
        py_embedding.cpp
        py_regular_graph.cpp
        py_tree_graph.cpp
        py_graph_misc.cpp
        py_lca_fast.cpp
        py_hierarchy_core.cpp
        py_pink_io.cpp)

set(MODULE_FILES
        __init__.py)

pybind11_add_module(higram pymodule.cpp ${PYMODULE_COMPONENTS})
set(UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} higram CACHE INTERNAL "" FORCE)


foreach (FILE ${MODULE_FILES})
    configure_file(${FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILE} COPYONLY)
endforeach ()