

find_package(Boost REQUIRED COMPONENTS unit_test_framework)
find_package(PythonInterp 3.5 REQUIRED)


include_directories(${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/lib/include ${PYTHON_INCLUDE_DIR})
# ${PROJECT_SOURCE_DIR}/lib/pybind11/include
#pybind11_add_module(xtensor_python_test example_xtensor.cpp)

set(PYMODULE_COMPONENTS
        py_undirected_graph.cpp
        py_embedding.cpp
        py_regular_graph.cpp
        py_tree_graph.cpp)

pybind11_add_module(higra pymodule.cpp ${PYMODULE_COMPONENTS})
set(UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} higra CACHE INTERNAL "" FORCE)