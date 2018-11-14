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
        test_alignment.cpp
        test_graph_core.cpp
        test_graph_weights.cpp
        test_horizontal_cuts.cpp
        test_rag.cpp
        test_tree.cpp
        test_watershed.cpp)

set(PATH_PREFIX algo/)

foreach (FILE ${FILES})
    set(TEST_CPP_COMPONENTS ${TEST_CPP_COMPONENTS}
            ${PATH_PREFIX}${FILE}
            CACHE INTERNAL "" FORCE)
endforeach ()