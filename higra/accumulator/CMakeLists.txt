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
        at_accumulator.py
        graph_accumulator.py
        tree_accumulator.py
        tree_contour_accumulator.py)

set(PYMODULE_COMPONENTS ${PYMODULE_COMPONENTS}
        ${CMAKE_CURRENT_SOURCE_DIR}/py_accumulators.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_at_accumulator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_graph_accumulator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_tree_accumulators.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_tree_contour_accumulator.cpp
        PARENT_SCOPE)

REGISTER_PYTHON_MODULE_FILES("${PY_FILES}")

