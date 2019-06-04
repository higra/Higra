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
        ${CMAKE_CURRENT_SOURCE_DIR}/test_dendrogram_purity.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_fragmentation_curve.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_partition.cpp
        PARENT_SCOPE)
