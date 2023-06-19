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
        ${CMAKE_CURRENT_SOURCE_DIR}/test_alignment.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_bipartite_graph.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_core.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_weights.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_horizontal_cuts.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_tree_energy_optimization.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_rag.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_tree.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_tree_fusion.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_tree_monotonic_regression.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_watershed.cpp
        PARENT_SCOPE)
