############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################



include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib/include ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIRS})

add_subdirectory(cpp)

set(MODULE_FILES
        __init__.py
        alignment.py
        assessment_fragmentation_curve.py
        attributes.py
        concept.py
        data_cache.py
        graph_core.py
        graph_image.py
        graph_weights.py
        hg_utils.py
        hierarchy_core.py
        image_utils.py
        pink_io.py
        rag.py
        tree_algo.py
        tree_accumulator.py
        tree_io.py
        watershed.py
        watershed_hierarchy.py)

pybind11_add_module(higram pymodule.cpp ${PYMODULE_COMPONENTS})
set(UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} higram CACHE INTERNAL "" FORCE)

foreach (FILE ${MODULE_FILES})
    configure_file(${FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILE} COPYONLY)
endforeach ()