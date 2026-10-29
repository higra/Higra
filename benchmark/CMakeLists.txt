############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

# Inherit the root C++17, optimization, LTO and SIMD settings for every method.
# Floating point validation/infinities require ordinary IEEE comparisons.
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
        utils.cpp
        benchmark_bpt_canonical.cpp
        #benchmark_lca.cpp
        #        benchmark_component_tree_casf.cpp
        #benchmark_undirected_graph.cpp
        #benchmark_regular_graph.cpp
        #benchmark_accumulator.cpp
        #benchmark_parallel_sort.cpp
        #benchmark_tree_iterator.cpp
        #benchmark_array_accessor.cpp
        #benchmark_views.cpp
        #benchmark_tree_attributes.cpp
        )

set(BENCHMARK_TARGET benchmark_higra)
add_executable(${BENCHMARK_TARGET} ${FILES_BENCHMARK})
foreach(target IN ITEMS ${BENCHMARK_TARGET})
    target_link_libraries(${target} PRIVATE benchmark::benchmark)
    if (HG_USE_TBB)
        target_compile_definitions(${target} PRIVATE HG_USE_TBB XTENSOR_USE_TBB)
        target_link_libraries(${target} PRIVATE TBB::tbb)
    endif()
    if (MSVC)
        target_compile_options(${target} PRIVATE /fp:precise)
    else()
        target_compile_options(${target} PRIVATE -fno-fast-math)
    endif()
endforeach()


add_custom_target(benchmark_exe
        COMMAND benchmark_higra
        DEPENDS ${BENCHMARK_TARGET})
