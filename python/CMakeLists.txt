

find_package(Boost REQUIRED COMPONENTS unit_test_framework)
include_directories(${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/lib/include)
# ${PROJECT_SOURCE_DIR}/lib/pybind11/include
pybind11_add_module(xtensor_python_test example_xtensor.cpp)
pybind11_add_module(example example.cpp)