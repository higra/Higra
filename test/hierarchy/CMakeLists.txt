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
        test_binary_partition_tree.cpp
        test_hierarchy_core.cpp
        test_watershed_hierarchy.cpp)

set(PATH_PREFIX hierarchy/)

foreach (FILE ${FILES})
    set(TEST_CPP_COMPONENTS ${TEST_CPP_COMPONENTS}
            ${PATH_PREFIX}${FILE}
            CACHE INTERNAL "" FORCE)
endforeach ()

