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
        ${CMAKE_CURRENT_SOURCE_DIR}/test_binary_partition_tree.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_component_tree.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_hierarchy_core.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_watershed_hierarchy.cpp
        PARENT_SCOPE)


