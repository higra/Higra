############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

add_subdirectory(resources)
set(PYTHON_MODULE_PATH ${CMAKE_BINARY_DIR}/python/)

set(PY_FILES
        test_concept.py
        test_data_cache.py
        test_hg_utils.py)

REGISTER_PYTHON_MODULE_FILES("${PY_FILES}")

add_subdirectory(test_accumulator)
add_subdirectory(test_algo)
add_subdirectory(test_assessment)
add_subdirectory(test_attribute)
add_subdirectory(test_hierarchy)
add_subdirectory(test_image)
add_subdirectory(test_interop)
add_subdirectory(test_io_utils)
add_subdirectory(test_structure)

add_test(NAME Test_python
COMMAND ${PYTHON_EXECUTABLE} -c "import sys;\
sys.path.insert(0, '${CMAKE_CURRENT_BINARY_DIR}/../..');\
import unittest;\
result=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('${CMAKE_CURRENT_BINARY_DIR}'));
exit(0 if result.wasSuccessful() else 1)"
        )
