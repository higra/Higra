############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################


set(FILES
        py_alignement.cpp
        py_graph_core.cpp
        py_graph_weights.cpp
        py_rag.cpp
        py_tree.cpp
        py_watershed.cpp)

set(PATH_PREFIX cpp/algo/)

foreach (FILE ${FILES})
    set(PYMODULE_COMPONENTS ${PYMODULE_COMPONENTS}
            ${PATH_PREFIX}${FILE}
            CACHE INTERNAL "" FORCE)
endforeach ()