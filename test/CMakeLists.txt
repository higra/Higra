############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################


if (DO_TEST)
    find_package(Boost REQUIRED COMPONENTS unit_test_framework)
    include_directories(${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib/include)

    set(HIGRA_TESTS_SRC
            test.cpp
            test_accumulator.cpp
            test_algo_tree.cpp
            test_binary_partition_tree.cpp
            test_embedding.cpp
            test_fibonacci_heap.cpp
            test_graph_image.cpp
            test_graph_weights.cpp
            test_hierarchy_core.cpp
            test_lca.cpp
            test_light_axis_view.cpp
            test_pink_graph_io.cpp
            test_point.cpp
            test_rag.cpp
            test_regular_graph.cpp
            test_tree.cpp
            test_tree_accumulator.cpp
            test_tree_io.cpp
            test_undirected_graph.cpp
            test_utils.cpp
            test_watershed.cpp)

    add_executable(test_exe ${HIGRA_TESTS_SRC})

    ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
    target_link_libraries(test_exe ${Boost_LIBRARIES})
    add_test(NAME Test_cpp COMMAND test_exe)
    set(UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_exe CACHE INTERNAL "" FORCE)

    add_subdirectory(ressources)

endif (DO_TEST)

