message(STATUS "Forcing tests build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ffast-math -Wunused-parameter -Wextra -Wreorder")
    CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)

    if (HAS_CPP14_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    else ()
        message(FATAL_ERROR "Unsupported compiler -- xtensor requires C++14 support!")
    endif ()

    # Enable link time optimization and set the default symbol
    # visibility to hidden (very important to obtain small binaries)
    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
        # Default symbol visibility
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

        # Check for Link Time Optimization support
        # (GCC/Clang)
        # LTO had to be removed as google benchmark doesn't build with it
        # CHECK_CXX_COMPILER_FLAG("-flto" HAS_LTO_FLAG)
        # if (HAS_LTO_FLAG)
        #     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
        # endif()

        # Intel equivalent to LTO is called IPO
        if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            CHECK_CXX_COMPILER_FLAG("-ipo" HAS_IPO_FLAG)
            if (HAS_IPO_FLAG)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo")
            endif ()
        endif ()
    endif ()
endif ()

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
    foreach (flag_var
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        string(REPLACE "/MD" "-MT" ${flag_var} "${${flag_var}}")
    endforeach ()
endif ()


find_package(benchmark REQUIRED)


include_directories(${PROJECT_SOURCE_DIR}/lib/include ${PROJECT_SOURCE_DIR}/include)

include_directories(${GBENCHMARK_INCLUDE_DIRS})

set(FILES_BENCHMARK
        main.cpp
        benchmark_tree_iterator.cpp
        )

set(BENCHMARK_TARGET benchmark_higra)
add_executable(${BENCHMARK_TARGET} ${FILES_BENCHMARK})
target_link_libraries(${BENCHMARK_TARGET} benchmark -lpthread)

add_custom_target(benchmark_exe
        COMMAND benchmarkx
        DEPENDS ${BENCHMARK_TARGET})