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
        alignment.py
        graph_core.py
        graph_weights.py
        horizontal_cuts.py
        rag.py
        tree.py
        tree_energy_optimization.py
        tree_fusion.py
        watershed.py)

set(PYMODULE_COMPONENTS ${PYMODULE_COMPONENTS}
        ${CMAKE_CURRENT_SOURCE_DIR}/py_alignement.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_graph_core.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_graph_weights.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_horizontal_cuts.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_rag.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_tree.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_tree_energy_optimization.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_tree_fusion.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_watershed.cpp
        PARENT_SCOPE)

REGISTER_PYTHON_MODULE_FILES("${PY_FILES}")