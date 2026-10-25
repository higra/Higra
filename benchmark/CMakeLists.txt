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
option(HG_BPT_BENCH_DIAGNOSTICS "Build separate BPT phase and CAS diagnostic runners." OFF)
add_executable(benchmark_bpt_grid benchmark_bpt_grid.cpp)
add_test(NAME Test_bpt_grid_runner COMMAND benchmark_bpt_grid --self-test)
set(BPT_BENCH_TARGETS ${BENCHMARK_TARGET} benchmark_bpt_grid)
if (HG_BPT_BENCH_DIAGNOSTICS)
    foreach(kind IN ITEMS diagnostics cas)
        add_executable(benchmark_higra_${kind} ${FILES_BENCHMARK})
        target_compile_definitions(benchmark_higra_${kind} PRIVATE HG_BPT_BENCH_DIAGNOSTICS)
        list(APPEND BPT_BENCH_TARGETS benchmark_higra_${kind})
    endforeach()
    target_compile_definitions(benchmark_higra_cas PRIVATE HG_BPT_BENCH_CAS)
endif()
foreach(target IN LISTS BPT_BENCH_TARGETS)
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
