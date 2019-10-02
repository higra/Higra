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
        graph_attributes.py
        tree_attributes.py)

set(PYMODULE_COMPONENTS ${PYMODULE_COMPONENTS}
        ${CMAKE_CURRENT_SOURCE_DIR}/py_tree_attributes.cpp
        PARENT_SCOPE)

REGISTER_PYTHON_MODULE_FILES("${PY_FILES}")





