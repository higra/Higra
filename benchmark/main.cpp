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
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_arena.h>

// Custom main function to restrict TBB threads
int main(int argc, char **argv) {
    std::size_t num_threads = oneapi::tbb::this_task_arena::max_concurrency();

    if (const char* env_p = std::getenv("TBB_NUM_THREADS")) {
        int parsed = std::atoi(env_p);
        if (parsed > 0) {
            num_threads = static_cast<std::size_t>(parsed);
        }
    }

    oneapi::tbb::global_control tbb_limit(
        oneapi::tbb::global_control::max_allowed_parallelism,
        num_threads
    );

    std::cout << "TBB parallelism constrained to: " << num_threads << " thread(s)\n\n";

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
}