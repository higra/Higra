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
        binary_partition_tree.py
        component_tree.py
        constrained_connectivity_hierarchy.py
        hierarchy_core.py
        watershed_hierarchy.py)

set(PYMODULE_COMPONENTS ${PYMODULE_COMPONENTS}
        ${CMAKE_CURRENT_SOURCE_DIR}/py_binary_partition_tree.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_component_tree.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_hierarchy_core.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_watershed_hierarchy.cpp
        PARENT_SCOPE)

REGISTER_PYTHON_MODULE_FILES("${PY_FILES}")
