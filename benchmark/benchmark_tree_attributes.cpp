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
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xgenerator.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xeval.hpp"
#include "higra/attribute/tree_attribute.hpp"

using namespace xt;
using namespace hg;

std::size_t min_tree_size = 10;
std::size_t max_tree_size = 16;

bool is_power_of_two(std::size_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

auto get_complete_binary_tree(std::size_t num_leaves) {
    assert(is_power_of_two(num_leaves));
    array_1d<std::size_t> parent = array_1d<std::size_t>::from_shape({num_leaves * 2 - 1});
    for (std::size_t i = 0, j = num_leaves; i < parent.size() - 1; j++) {
        parent(i++) = j;
        parent(i++) = j;
    }
    parent(parent.size() - 1) = parent.size() - 1;
    return tree(parent);
}

static void BM_tree_volume_cstyle(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        std::size_t size = state.range(0);
        xt::random::seed(42);
        auto t = get_complete_binary_tree(size);
        auto area = eval(xt::ones<double>({t.num_vertices()}));
        auto altitude = eval(xt::arange<double>(t.num_vertices())/t.num_vertices()*255.);
        state.ResumeTiming();
        array_1d<double> volume = empty<double>({t.num_vertices()});
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

static void BM_tree_volume_xtstyle(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        std::size_t size = state.range(0);
        xt::random::seed(42);
        auto t = get_complete_binary_tree(size);
        auto area = eval(xt::ones<double>({t.num_vertices()}));
        auto altitude = eval(xt::arange<double>(t.num_vertices())/t.num_vertices()*255.);
        state.ResumeTiming();

        auto node_height = xt::abs(altitude - xt::index_view(altitude, t.parents()));
        auto node_partial_volume = node_height * area;
        auto leave_volume = xt::view(node_partial_volume, xt::range(0, num_leaves(t)));
        auto volume = accumulate_and_combine_sequential(t, node_partial_volume, leave_volume, accumulator_sum(),
                                                 std::plus<double>());
        benchmark::DoNotOptimize(volume[t.root()]);
    }
}

BENCHMARK(BM_tree_volume_xtstyle)->Range(1 << min_tree_size, 1 << max_tree_size);