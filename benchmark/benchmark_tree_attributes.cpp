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
#include "higra/attribute/tree_attribute.hpp"
#include "utils.h"
#include <cmath>
#include <functional>

namespace benchmark_tree_attributes {

    using namespace hg;

    std::size_t min_tree_size = 10;
    std::size_t max_tree_size = 16;

    static void BM_tree_volume_cstyle(benchmark::State &state) {
        for (auto _ : state) {
            state.PauseTiming();

            std::size_t size = state.range(0);
            bench_utils::random_seed(42);
            auto t = get_complete_binary_tree(size);
            auto area = hg::ones<double>({t.num_vertices()});
            const double n = (double) t.num_vertices();
            auto altitude = hg::map([n](double i) { return i / n * 255.; }, hg::arange<double>(t.num_vertices()));
            t.compute_children();
            state.ResumeTiming();
            array_1d<double> volume = hg::empty<double>({t.num_vertices()});
            auto & parent = t.parents();
            for (auto i: leaves_to_root_iterator(t)) {
                volume(i) = std::fabs(altitude(i) - altitude(parent(i))) * area(i);
                for (auto c: t.children(i)) {
                    volume[i] += volume[c];
                }
            }

            benchmark::DoNotOptimize(volume[t.root()]);
        }
    }

    BENCHMARK(BM_tree_volume_cstyle)->Range(1 << min_tree_size, 1 << max_tree_size);

    // Whole-array style: elementwise operations with hg::map, then an accumulator.
    static void BM_tree_volume_mapstyle(benchmark::State &state) {
        for (auto _ : state) {
            state.PauseTiming();

            std::size_t size = state.range(0);
            bench_utils::random_seed(42);
            auto t = get_complete_binary_tree(size);
            auto area = hg::ones<double>({t.num_vertices()});
            const double n = (double) t.num_vertices();
            auto altitude = hg::map([n](double i) { return i / n * 255.; }, hg::arange<double>(t.num_vertices()));
            state.ResumeTiming();

            auto node_height = hg::map([](double a, double b) { return std::fabs(a - b); },
                                       altitude, hg::take(altitude, t.parents()));
            auto node_partial_volume = hg::map(std::multiplies<double>(), node_height, area);
            auto leave_volume = node_partial_volume.slice(0, num_leaves(t));
            auto volume = accumulate_and_combine_sequential(t, node_partial_volume, leave_volume, accumulator_sum(),
                                                     std::plus<double>());
            benchmark::DoNotOptimize(volume[t.root()]);
        }
    }

    BENCHMARK(BM_tree_volume_mapstyle)->Range(1 << min_tree_size, 1 << max_tree_size);
}
