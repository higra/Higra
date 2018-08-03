############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

add_subdirectory(ressources)
set(PYTHON_MODULE_PATH ${CMAKE_BINARY_DIR}/python/)

file(GLOB PYTHON_TEST_FILES
        "*.py"
        )

foreach (FILE ${PYTHON_TEST_FILES})
    get_filename_component(filename ${FILE} NAME)
    configure_file(${filename} ${CMAKE_CURRENT_BINARY_DIR}/${filename})


endforeach ()


add_test(NAME Test_python
COMMAND ${PYTHON_EXECUTABLE} -c "import sys;\
sys.path.insert(0, '${CMAKE_CURRENT_BINARY_DIR}/..');\
import unittest;\
result=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('${CMAKE_CURRENT_BINARY_DIR}'));
exit(0 if result.wasSuccessful() else 1)"
        )
#${CMAKE_CURRENT_BINARY_DIR} import os;print(os.getcwd());