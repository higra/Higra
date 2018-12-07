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

##########################
#  VERSION
##########################

file(STRINGS "include/higra/config.hpp" higra_version_defines
        REGEX "#define HIGRA_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${higra_version_defines})
    if(ver MATCHES "#define HIGRA_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(HIGRA_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION
        ${HIGRA_VERSION_MAJOR}.${HIGRA_VERSION_MINOR}.${HIGRA_VERSION_PATCH})
message(STATUS "Building higra v${${PROJECT_NAME}_VERSION}")

##########################
#  CMAKE config
##########################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/lib/")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_SOURCE_DIR}/lib/")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(higraTools)

##########################
#  Build Type and flags
##########################

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."  FORCE)
endif(NOT CMAKE_BUILD_TYPE)

include(CheckCXXCompilerFlag)
message("${CMAKE_CXX_COMPILER_ID}")

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-local-typedefs")# -Wno-missing-braces -Wno-mismatched-tags -Wno-unneeded-internal-declaration")
    CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
    if(COMPILER_SUPPORTS_CXX14)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    else()
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
    endif()

    # Not debug => optim flag
    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -ffast-math -Wno-strict-overflow")#-march=native

        # check link time optimization -faligned-new
        CHECK_CXX_COMPILER_FLAG("-flto" HAS_LTO_FLAG)
        if (HAS_LTO_FLAG)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
        endif()
        if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            CHECK_CXX_COMPILER_FLAG("-ipo" HAS_IPO_FLAG)
            if (HAS_IPO_FLAG)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo")
            endif ()
        endif ()
    endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # using Visual Studio C++
    CHECK_CXX_COMPILER_FLAG("/std:c++14" COMPILER_SUPPORTS_CXX14)
    if(COMPILER_SUPPORTS_CXX14)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++14")
    else()
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
    endif()

    set(Boost_USE_STATIC_LIBS ON)

    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast /GL /EHsc /MP /bigobj /O2")
        set(CMAKE_EXE_LINKER_FLAGS /LTCG /MANIFEST:NO)
    endif()
endif()

##########################
#  Options
##########################

find_package(Boost 1.4 QUIET COMPONENTS unit_test_framework)
if (Boost_FOUND)
    option(DO_TEST
            "Execute test suit." ON)
else ()
    message("Testing disabled as 'Boost unit test framework' cannot be found. Please install it or manually provide its location to cmake in order to enable testing.")
    option(DO_TEST
            "Execute test suit." OFF)
endif () 

enable_testing()

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
    add_definitions("-DHG_ENABLE_TRACE")
endif ()


##########################
#  Build Python bindings
##########################

#set(PYBIND11_CPP_STANDARD -std=c++14)
add_subdirectory(python)

##########################
#  Build Tests
##########################

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

##########################
#  Build benchmark
##########################

if (DO_BENCHMARK)
    add_subdirectory(benchmark)
endif (DO_BENCHMARK)