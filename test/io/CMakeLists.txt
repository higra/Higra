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
        test_pink_graph_io.cpp
        test_pnm_io.cpp
        test_tree_io.cpp)

set(PATH_PREFIX io/)

foreach (FILE ${FILES})
    set(TEST_CPP_COMPONENTS ${TEST_CPP_COMPONENTS}
            ${PATH_PREFIX}${FILE}
            PARENT_SCOPE)
endforeach ()

