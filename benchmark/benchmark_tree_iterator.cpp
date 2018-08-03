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

static void BM_tree_accumulate_parallel_scalar_cstyle(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        std::size_t size = state.range(0);
        xt::random::seed(42);
        auto t = get_complete_binary_tree(size);
        auto input = eval(random::randn<double>({t.num_vertices()}));
        state.ResumeTiming();
        array_nd<double> output = array_nd<double>::from_shape({t.num_vertices()});
        auto sout = output.data();
        auto sin = input.data();
        std::fill(sout, sout + t.num_vertices(), 0);
        for (auto i: t.iterate_from_leaves_to_root(leaves_it::exclude)) {
            for (auto c: t.children(i)) {
                sout[i] += sin[c];
            }
        }

        benchmark::DoNotOptimize(sout[t.root()]);
    }
}

BENCHMARK(BM_tree_accumulate_parallel_scalar_cstyle)->Range(1 << min_tree_size, 1 << max_tree_size);

/*
static void BM_tree_accumulate_parallel_scalar_view(benchmark::State& state) {
    for (auto _ : state)
    {
        state.PauseTiming();

        std::size_t size = state.range(0);
        xt::random::seed(42);
        auto t = get_complete_binary_tree(size);
        auto input = eval(random::randn<double>({t.num_vertices()}));
        state.ResumeTiming();
        array_nd <double> output;
        benchmark::DoNotOptimize(output = accumulate_parallel(t, input, accumulator_sum<double>()));

    }
}

BENCHMARK(BM_tree_accumulate_parallel_scalar)->Range(1<<min_tree_size, 1<<max_tree_size);
*/
static void BM_tree_accumulate_parallel_light_view_scalar(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        std::size_t size = state.range(0);
        xt::random::seed(42);
        auto t = get_complete_binary_tree(size);
        auto input = eval(random::randn<double>({t.num_vertices()}));
        state.ResumeTiming();
        array_nd<double> output;
        benchmark::DoNotOptimize(output = accumulate_parallel(t, input, accumulator_sum()));

    }
}

BENCHMARK(BM_tree_accumulate_parallel_light_view_scalar)->Range(1 << min_tree_size, 1 << max_tree_size);

static void BM_tree_accumulate_parallel_light_view_vectorial(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        std::size_t size = state.range(0);
        xt::random::seed(42);
        auto t = get_complete_binary_tree(size);
        auto input = eval(random::randn<double>({t.num_vertices(), 3ul, 3ul}));
        state.ResumeTiming();
        array_nd<double> output;
        benchmark::DoNotOptimize(output = accumulate_parallel(t, input, accumulator_sum()));

    }
}

BENCHMARK(BM_tree_accumulate_parallel_light_view_vectorial)->Range(1 << min_tree_size, 1 << max_tree_size);
/*
static void BM_tree_accumulate_parallel_vectorial(benchmark::State& state) {
    for (auto _ : state)
    {
        state.PauseTiming();

        std::size_t size = state.range(0);
        xt::random::seed(42);
        auto t = get_complete_binary_tree(size);
        auto input = eval(random::randn<double>({t.num_vertices(), 3ul, 3ul}));
        state.ResumeTiming();
        array_nd <double> output;
        benchmark::DoNotOptimize(output = accumulate_parallel(t, input, accumulator_sum<double>()));

    }
}

BENCHMARK(BM_tree_accumulate_parallel_vectorial)->Range(1<<min_tree_size, 1<<max_tree_size);
 */