


if (DO_TEST)
    find_package(Boost REQUIRED COMPONENTS unit_test_framework)
    include_directories(${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/lib/include)

    set(HIGRA_TESTS_SRC
            test.cpp
            test_xtensor.cpp
            test_imageio.cpp
            test_undirected_graph.cpp
            test_embedding.cpp
            test_utils.cpp
            test_regular_graph.cpp
            test_tree.cpp
            test_point.cpp
            test_graph_misc.cpp)


    add_executable(test_exe ${HIGRA_TESTS_SRC})

    TARGET_LINK_LIBRARIES (test_exe ${OpenCV_LIBS})

    ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
    target_link_libraries(test_exe ${Boost_LIBRARIES})
    add_test(NAME Test_cpp COMMAND test_exe)
    set(UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_exe CACHE INTERNAL "" FORCE)
endif (DO_TEST)

add_subdirectory(python)