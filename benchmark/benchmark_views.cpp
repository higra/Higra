/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <benchmark/benchmark.h>

#include "higra/graph.hpp"
#include "higra/accumulator/tree_accumulator.hpp"
#include "utils.h"

namespace benchmark_views {

    using namespace hg;

    std::size_t min_tree_size = 10;
    std::size_t max_tree_size = 16;

    static void BM_tree_propagate_parallel(benchmark::State &state) {
        for (auto _ : state) {
            state.PauseTiming();

            std::size_t size = state.range(0);
            bench_utils::random_seed(42);
            auto t = get_complete_binary_tree(size);
            auto input = bench_utils::random_normal<double>({t.num_vertices()});
            state.ResumeTiming();
            array_nd<double> output = array_nd<double>::from_shape({t.num_vertices()});
            auto &parents = t.parents();
            for (auto i: leaves_to_root_iterator(t, leaves_it::exclude)) {
                output[i] = input[parents[i]];
            }

            benchmark::DoNotOptimize(output[t.root()]);
        }
    }

    BENCHMARK(BM_tree_propagate_parallel)->Range(1 << min_tree_size, 1 << max_tree_size);

    // Same propagation as a gather of the parent values (hg::take)
    static void BM_take_propagate_parallel(benchmark::State &state) {
        for (auto _ : state) {
            state.PauseTiming();

            std::size_t size = state.range(0);
            bench_utils::random_seed(42);
            auto t = get_complete_binary_tree(size);
            auto input = bench_utils::random_normal<double>({t.num_vertices()});
            state.ResumeTiming();
            array_nd<double> output = hg::take(input, t.parents());

            benchmark::DoNotOptimize(output[t.root()]);
        }
    }

    BENCHMARK(BM_take_propagate_parallel)->Range(1 << min_tree_size, 1 << max_tree_size);
}
