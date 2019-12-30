############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

set(TEST_CPP_COMPONENTS ${TEST_CPP_COMPONENTS}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_embedding.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_fibonacci_heap.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_lca.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_point.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_regular_graph.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_tree.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_undirected_graph.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/details/test_iterator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/details/test_light_axis_view.cpp
        PARENT_SCOPE)



