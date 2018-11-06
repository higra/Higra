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
        py_embedding.cpp
        py_lca_fast.cpp
        py_regular_graph.cpp
        py_tree_graph.cpp
        py_undirected_graph.cpp)

set(PATH_PREFIX cpp/structure/)

foreach (FILE ${FILES})
    set(PYMODULE_COMPONENTS ${PYMODULE_COMPONENTS}
            ${PATH_PREFIX}${FILE}
            CACHE INTERNAL "" FORCE)
endforeach ()