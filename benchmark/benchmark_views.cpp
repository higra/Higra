//
// Created by perretb on 19/07/18.
//

//
// Created by user on 5/11/18.
//

#include <benchmark/benchmark.h>

#include "higra/graph.hpp"
#include "higra/accumulator/tree_accumulator.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xgenerator.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xeval.hpp"

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

static void BM_tree_propagate_parallel(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        std::size_t size = state.range(0);
        xt::random::seed(42);
        auto t = get_complete_binary_tree(size);
        auto input = eval(random::randn<double>({t.num_vertices()}));
        state.ResumeTiming();
        array_nd<double> output = array_nd<double>::from_shape({t.num_vertices()});
        auto &parents = t.parents();
        for (auto i: leaves_to_root_iterator(t, leaves_it::exclude)) {
            output.unchecked(i) = input.unchecked(parents.unchecked(i));
        }

        benchmark::DoNotOptimize(output[t.root()]);
    }
}

BENCHMARK(BM_tree_propagate_parallel)->Range(1 << min_tree_size, 1 << max_tree_size);

static void BM_view_propagate_parallel(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        std::size_t size = state.range(0);
        xt::random::seed(42);
        auto t = get_complete_binary_tree(size);
        auto input = eval(random::randn<double>({t.num_vertices()}));
        state.ResumeTiming();
        array_nd<double> output = xt::eval(xt::index_view(input, t.parents()));

        benchmark::DoNotOptimize(output[t.root()]);
    }
}

BENCHMARK(BM_view_propagate_parallel)->Range(1 << min_tree_size, 1 << max_tree_size);