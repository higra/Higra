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
find_package(pybind11 REQUIRED)
find_package(NumPy REQUIRED)

add_subdirectory(higra)
add_subdirectory(test)

if (DO_TEST)
    add_subdirectory(test_cpp)
endif()

configure_file(setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(MANIFEST.in ${CMAKE_CURRENT_BINARY_DIR}/MANIFEST)
configure_file(README.md ${CMAKE_CURRENT_BINARY_DIR}/README.md COPYONLY)