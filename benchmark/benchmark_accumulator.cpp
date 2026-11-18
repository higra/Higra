/***************************************************************************
* Copyright ESIEE Paris (2021)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/


#include <benchmark/benchmark.h>
#include "utils.h"
#include "higra/structure/array.hpp"
#include "higra/accumulator/tree_accumulator.hpp"

using namespace hg;


std::size_t min_tree_size = 10;
std::size_t max_tree_size = 20;


static void BM_tree_accumulator(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        std::size_t size = state.range(0);
        bench_utils::random_seed(42);

        auto t = get_complete_binary_tree(size);
        array_1d<int> area = hg::ones<int>({t.num_leaves()});

        state.ResumeTiming();
        auto res = hg::accumulate_sequential(t, area, hg::accumulator_sum());

        benchmark::DoNotOptimize(res[t.root()]);
    }
}

BENCHMARK(BM_tree_accumulator)->Range(1 << min_tree_size, 1 << max_tree_size);





