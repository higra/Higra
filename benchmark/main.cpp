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
#include <stdexcept>
#include <string>

#include <benchmark/benchmark.h>
#ifdef HG_USE_TBB
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/version.h>
#endif

#ifdef HG_BENCHMARK_BPT
bool bpt_single_run(int argc, char **argv, std::size_t thread_limit);
#endif

// Custom main function to restrict TBB threads
int main(int argc, char **argv) {
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
#ifdef HG_BENCHMARK_BPT
    if (bpt_single_run(argc, argv, num_threads)) return 0;
#endif

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
