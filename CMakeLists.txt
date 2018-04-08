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

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

#set(CMAKE_CXX_STANDARD 14)

#set(EXECUTABLE_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})




option (DO_TEST
        "Execute test suit." ON)

if (DO_TEST)
    enable_testing()
endif(DO_TEST)


option (USE_SIMD
        "Enabled SIMD support for xtensor." ON)

if (USE_SIMD)
    add_definitions("-DXTENSOR_USE_XSIMD")
endif(USE_SIMD)


FIND_PACKAGE( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )



#add_subdirectory(${PROJECT_SOURCE_DIR}/src)

set(PYBIND11_CPP_STANDARD -std=c++14)
add_subdirectory(lib/pybind11)

add_subdirectory(python/higra)


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
