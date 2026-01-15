############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.10)
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

#if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 3.12)
#    cmake_policy(SET CMP0074 NEW) # use XXX_ROOT env variable for find_package
#endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/lib/")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_SOURCE_DIR}/lib/")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/tools")
include(higraTools)

##########################
#  Build Type and flags
##########################

# new build type coverage
message("* Adding build types...")
SET(CMAKE_CXX_FLAGS_COVERAGE
        "${GCC_DEBUG_FLAGS} -fprofile-arcs -ftest-coverage -fprofile-update=atomic"
        CACHE STRING "Flags used by the C++ compiler during coverage builds."
        FORCE )
SET(CMAKE_C_FLAGS_COVERAGE
        "${GCC_DEBUG_FLAGS} -fprofile-arcs -ftest-coverage -fprofile-update=atomic"
        CACHE STRING "Flags used by the C compiler during coverage builds."
        FORCE )
SET(CMAKE_EXE_LINKER_FLAGS_COVERAGE
        ""
        CACHE STRING "Flags used for linking binaries during coverage builds."
        FORCE )
SET(CMAKE_SHARED_LINKER_FLAGS_COVERAGE
        ""
        CACHE STRING "Flags used by the shared libraries linker during coverage builds."
        FORCE )
MARK_AS_ADVANCED(
        CMAKE_CXX_FLAGS_COVERAGE
        CMAKE_C_FLAGS_COVERAGE
        CMAKE_EXE_LINKER_FLAGS_COVERAGE
        CMAKE_SHARED_LINKER_FLAGS_COVERAGE )

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel Coverage."  FORCE)
endif()
message("Current build type is: ${CMAKE_BUILD_TYPE}")

include(CheckCXXCompilerFlag)
message("Current compiler is: ${CMAKE_CXX_COMPILER_ID}")

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

find_program(CCACHE_FOUND "ccache")
if(CCACHE_FOUND)
    message("Using ccache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif(CCACHE_FOUND)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wno-error=changes-meaning) # to avoid error with recent gcc versions and TBB, to be removed when TBB is updated

if ((CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT (CMAKE_CXX_SIMULATE_ID MATCHES "MSVC") ) OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-local-typedefs")# -Wno-unused-but-set-variable -Wno-missing-braces -Wno-mismatched-tags -Wno-unneeded-internal-declaration")

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
    endif()

    CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
    if(COMPILER_SUPPORTS_CXX17)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    else()
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
    endif()

    # Not debug => optim flag
    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-overflow")#-march=native

        # check link time optimization -faligned-new
        CHECK_CXX_COMPILER_FLAG("-flto" HAS_LTO_FLAG)
        if (HAS_LTO_FLAG)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=auto")
        endif()
        if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            CHECK_CXX_COMPILER_FLAG("-ipo" HAS_IPO_FLAG)
            if (HAS_IPO_FLAG)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo")
            endif ()
        endif ()
    endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC")) 
    # @todo Clang as MVSC:more works needed for flags 
    # using Visual Studio C++
    CHECK_CXX_COMPILER_FLAG("/std:c++17" COMPILER_SUPPORTS_CXX17)
    if(COMPILER_SUPPORTS_CXX17)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
    else()
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
    endif()

    add_definitions("-DNOMINMAX")
    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  /GL /EHsc /MP /bigobj /O2 /wd4244 /wd4267 /Zc:lambda")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG /MANIFEST:NO")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  /EHsc /MP /bigobj /wd4244 /wd4267 /Zc:lambda")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}  /MANIFEST:NO")
    endif()
else()
    message(FATAL_ERROR "Unknown compiler, please use GCC, Clang or MSVC.")
endif()


##########################
#  Options
##########################

find_package(Catch2 REQUIRED)
option(DO_CPP_TEST
        "Build c++ test suit." ON)

option(DO_EMBEDDED_PYTHON_CPP_TEST
        "Build c++ test with python interpreter embedded suit." OFF)

enable_testing()

option (USE_SIMD
        "Enabled SIMD support for xtensor." ON)

if (USE_SIMD)
    add_definitions("-DXTENSOR_USE_XSIMD")
endif()

option(DO_BENCHMARK
        "Build benchmark suit." OFF)
if (DO_BENCHMARK)
    if(${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
        message(FATAL_ERROR "Benchmark cannot be built in Debug mode")
    endif()
    find_package(benchmark REQUIRED)
endif ()

if(NOT ${U_CMAKE_BUILD_TYPE} MATCHES RELEASE)
    add_definitions("-DHG_ENABLE_TRACE")
endif()

option(HG_USE_TBB
        "Enable Intel TBB support." OFF)

if (HG_USE_TBB)
    find_package(TBB REQUIRED)
    message(STATUS "Found intel TBB: ${TBB_INCLUDE_DIRS}")
    add_definitions("-DTBB_SUPPRESS_DEPRECATED_MESSAGES")
endif ()

option(HG_BUILD_WHEEL
        "Should be set to On when building a wheel." OFF)

option(HG_UNITY_BUILD
        "Enable unity build." OFF)

set(HG_UNITY_BUILD_BATCH_SIZE 8 CACHE STRING "Set the batch size for unity build.")
string(REGEX MATCH "^[0-9]+$" IS_VALID_HG_UNITY_BUILD_BATCH_SIZE "${HG_UNITY_BUILD_BATCH_SIZE}")
if(NOT IS_VALID_HG_UNITY_BUILD_BATCH_SIZE)
    message(FATAL_ERROR "HG_UNITY_BUILD_BATCH_SIZE must be an integer. Provided: ${HG_UNITY_BUILD_BATCH_SIZE}")
else()
    message(STATUS "Unity build batch size: ${HG_UNITY_BUILD_BATCH_SIZE}")
endif()


##########################
#  Build Python bindings
##########################

find_package(Python COMPONENTS Interpreter NumPy REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(higra)
configure_file(setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(README.md ${CMAKE_CURRENT_BINARY_DIR}/README.md COPYONLY)

##########################
#  Build Tests
##########################
option(DO_AUTO_TEST
        "Execute test suit automatically at the end of the build." OFF)
add_subdirectory(test)

##########################
#  Build benchmark
##########################

if (DO_BENCHMARK)
    add_subdirectory(benchmark)
endif ()

message("CMAKE_CXX_FLAGS" ${CMAKE_CXX_FLAGS})
message("CMAKE_EXE_LINKER_FLAGS" ${CMAKE_EXE_LINKER_FLAGS})

