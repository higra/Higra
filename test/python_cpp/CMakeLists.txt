############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################


include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib/include ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})

set(HIGRA_TESTS_PYTHON_SRC
        main.cpp)

add_executable(test_python_exe ${HIGRA_TESTS_PYTHON_SRC})

target_link_libraries(test_python_exe
        PRIVATE
        pybind11::embed
        Catch2::Catch2WithMain)


add_test(NAME Test_python_cpp COMMAND test_python_exe)
set(UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_python_exe PARENT_SCOPE)





