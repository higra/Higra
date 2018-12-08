############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

add_subdirectory(python)

if (DO_CPP_TEST)
    add_subdirectory(cpp)
    add_subdirectory(python_cpp)
    add_custom_target( all_tests ALL
            DEPENDS ${UNIT_TEST_TARGETS}
            )

    add_custom_command(TARGET all_tests
            COMMENT "Run tests"
            POST_BUILD COMMAND ctest
            #POST_BUILD COMMAND ctest ARGS --output-on-failure
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            )
endif(DO_CPP_TEST)
