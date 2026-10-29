/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <iostream>
#include <cstdlib>

#include <benchmark/benchmark.h>
#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xtensor.hpp"
#ifdef HG_USE_TBB
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/version.h>
#endif

bool bpt_single_run(int argc, char **argv, std::size_t thread_limit);

#ifdef XTENSOR_USE_XSIMD
#ifdef __GNUC__

template<class T>
void print_type(T && /*t*/) {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
}

#endif

void print_stats() {
    std::cout << "USING XSIMD\nSIMD SIZE: " << xsimd::simd_traits<double>::size << "\n\n";
#ifdef __GNUC__
    print_type(xt::xarray<double>());
    print_type(xt::xtensor<double, 2>());
#endif
}

#else

void print_stats() {
    std::cout << "NOT USING XSIMD\n\n";
};
#endif


// Custom main function to print SIMD config and restrict TBB threads
int main(int argc, char **argv) {
    // Single-call output is machine-readable.
    const bool bpt_command = argc > 1 && std::string(argv[1]).rfind("--bpt-", 0) == 0;
    if (!bpt_command) print_stats();
#ifdef HG_USE_TBB

    std::size_t num_threads = oneapi::tbb::this_task_arena::max_concurrency();

    if (const char* env_p = std::getenv("TBB_NUM_THREADS")) {
        std::size_t consumed = 0;
        const auto parsed = std::stoull(env_p, &consumed);
        if (consumed != std::string(env_p).size() || parsed == 0 || env_p[0] == '-') {
            throw std::invalid_argument("TBB_NUM_THREADS must be a positive integer.");
        }
        num_threads = parsed;
    }

    oneapi::tbb::global_control tbb_limit(
        oneapi::tbb::global_control::max_allowed_parallelism,
        num_threads
    );

    benchmark::AddCustomContext("tbb", "on");
    benchmark::AddCustomContext("tbb_version", TBB_VERSION_STRING);
    benchmark::AddCustomContext("tbb_interface", std::to_string(TBB_INTERFACE_VERSION));
    benchmark::AddCustomContext("active_thread_limit", std::to_string(oneapi::tbb::global_control::active_value(
        oneapi::tbb::global_control::max_allowed_parallelism)));
#else
    const std::size_t num_threads = 1;
    benchmark::AddCustomContext("tbb", "off");
#endif
    benchmark::AddCustomContext("thread_limit", std::to_string(num_threads));
    if (bpt_single_run(argc, argv, num_threads)) return 0;

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}