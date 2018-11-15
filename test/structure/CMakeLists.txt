############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

add_subdirectory(details)

set(FILES
        test_embedding.cpp
        test_fibonacci_heap.cpp
        test_lca.cpp
        test_point.cpp
        test_regular_graph.cpp
        test_tree.cpp
        test_undirected_graph.cpp)

set(PATH_PREFIX structure/)

foreach (FILE ${FILES})
    set(TEST_CPP_COMPONENTS ${TEST_CPP_COMPONENTS}
            ${PATH_PREFIX}${FILE}
            PARENT_SCOPE)
endforeach ()

