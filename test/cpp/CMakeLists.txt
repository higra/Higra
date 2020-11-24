############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################


if (DO_CPP_TEST)
    include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/lib/include)

    set(TEST_CPP_COMPONENTS ${TEST_CPP_COMPONENTS}
            test.cpp
            test_sorting.cpp
            test_utils.cpp)

    add_subdirectory(accumulator)
    add_subdirectory(algo)
    add_subdirectory(assessment)
    add_subdirectory(attribute)
    add_subdirectory(detail)
    add_subdirectory(hierarchy)
    add_subdirectory(image)
    add_subdirectory(io)
    add_subdirectory(structure)

    add_executable(test_exe ${TEST_CPP_COMPONENTS})

    if (HG_USE_TBB)
        add_definitions("-DXTENSOR_USE_TBB")
        target_compile_definitions(test_exe PRIVATE HG_USE_TBB)
        target_include_directories(test_exe PRIVATE ${TBB_INCLUDE_DIRS})
        target_link_libraries(test_exe PRIVATE ${TBB_LIBRARIES})
    endif ()

    #target_link_libraries(test_exe Catch2::Catch2)
    add_test(NAME Test_cpp COMMAND test_exe)
    set(UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_exe PARENT_SCOPE)

    add_subdirectory(resources)

endif (DO_CPP_TEST)

