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
        contour_2d.py
        graph_image.py
        hierarchy_mean_pb.py
        image_utils.py
        tree_of_shapes.py
        )

set(PYMODULE_COMPONENTS ${PYMODULE_COMPONENTS}
        ${CMAKE_CURRENT_SOURCE_DIR}/py_contour_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_graph_image.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_hierarchy_mean_pb.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/py_tree_of_shapes.cpp
        PARENT_SCOPE)

REGISTER_PYTHON_MODULE_FILES("${PY_FILES}")