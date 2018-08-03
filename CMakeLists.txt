############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 2.7)
project(higra)


set (HIGRA_VERSION_MAJOR 1 CACHE INTERNAL "" FORCE)
set (HIGRA_VERSION_MINOR 0 CACHE INTERNAL "" FORCE)
set (HIGRA_RELEASE_TYPE "alpha" CACHE INTERNAL "" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."  FORCE)
endif(NOT CMAKE_BUILD_TYPE)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++1y" COMPILER_SUPPORTS_CXX1Y)
if(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
elseif(COMPILER_SUPPORTS_CXX1Y)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

message("${CMAKE_CXX_COMPILER_ID}")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-local-typedefs -Wno-missing-braces -Wno-mismatched-tags -Wno-unneeded-internal-declaration")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-local-typedefs -Wno-strict-overflow")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++
endif()


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/lib/cmake/")

find_package(Boost QUIET COMPONENTS unit_test_framework)
if (Boost_FOUND)
    option(DO_TEST
            "Execute test suit." ON)
else ()
    message("Testing disabled as 'Boost unit test framework' cannot be found. Please install it or manually provide its location to cmake in order to enable testing.")
    option(DO_TEST
            "Execute test suit." OFF)
endif ()

if (DO_TEST)
    enable_testing()
endif(DO_TEST)


option (USE_SIMD
        "Enabled SIMD support for xtensor." ON)

if (USE_SIMD)
    add_definitions("-DXTENSOR_USE_XSIMD")
endif(USE_SIMD)


if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    find_package(benchmark QUIET)
    if (benchmark_FOUND)
        option(DO_BENCHMARK
                "Execute benchmark suit." ON)
    else ()
        message("Benchmarking disabled as 'Google benchmark' cannot be found. Please install it or manually provide its location to cmake in order to enable testing.")
        option(DO_BENCHMARK
                "Execute benchmark suit." OFF)
    endif ()
else ()
    message("Benchmarking disabled in 'debug' mode.")
    option(DO_BENCHMARK
            "Execute benchmark suit." OFF)
endif ()


#add_subdirectory(${PROJECT_SOURCE_DIR}/src)

set(PYBIND11_CPP_STANDARD -std=c++14)
add_subdirectory(lib/pybind11)

add_subdirectory(python)


if (DO_TEST)
    add_subdirectory(${PROJECT_SOURCE_DIR}/test)
    add_custom_target( all_tests ALL
            DEPENDS ${UNIT_TEST_TARGETS}
            )

    add_custom_command(TARGET all_tests
            COMMENT "Run tests"
            POST_BUILD COMMAND ctest
            #POST_BUILD COMMAND ctest ARGS --output-on-failure
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            )
endif(DO_TEST)


if (DO_BENCHMARK)
    add_subdirectory(benchmark)
endif (DO_BENCHMARK)