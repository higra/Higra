

find_package(Boost REQUIRED COMPONENTS unit_test_framework)
include_directories(${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/lib/include)
# ${PROJECT_SOURCE_DIR}/lib/pybind11/include
#pybind11_add_module(xtensor_python_test example_xtensor.cpp)

set(PYMODULE_COMPONENTS
        py_undirected_graph.cpp)

pybind11_add_module(higra pymodule.cpp ${PYMODULE_COMPONENTS})