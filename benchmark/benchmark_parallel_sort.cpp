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
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
#include <algorithm>
#include "tbb/parallel_sort.h"
#include "tbb-ssort/parallel_stable_sort.h"

using namespace xt;
using namespace hg;

std::size_t min_array_size = 10;
std::size_t max_array_size = 24;




static void BM_stl_sort(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        size_t size = state.range(0);
        array_1d<float> a = xt::random::rand<float>({size});
        state.ResumeTiming();
        std::sort(a.begin(), a.end());
        bool flag;
        benchmark::DoNotOptimize(flag = (a.size() == size));
    }
}

BENCHMARK(BM_stl_sort)->Range(1 << min_array_size, 1 << max_array_size);

static void BM_stl_stable_sort(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        size_t size = state.range(0);
        array_1d<float> a = xt::random::rand<float>({size});
        state.ResumeTiming();
        std::stable_sort(a.begin(), a.end());
        bool flag;
        benchmark::DoNotOptimize(flag = (a.size() == size));
    }
}

BENCHMARK(BM_stl_stable_sort)->Range(1 << min_array_size, 1 << max_array_size);

static void BM_tbb_parallel_sort(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        size_t size = state.range(0);
        array_1d<float> a = xt::random::rand<float>({size});
        state.ResumeTiming();
        tbb::parallel_sort(a.begin(), a.end());
        bool flag;
        benchmark::DoNotOptimize(flag = (a.size() == size));
    }
}

BENCHMARK(BM_tbb_parallel_sort)->Range(1 << min_array_size, 1 << max_array_size);

static void BM_tbb_parallel_stable_sort(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        size_t size = state.range(0);
        array_1d<float> a = xt::random::rand<float>({size});
        state.ResumeTiming();
        pss::parallel_stable_sort(a.begin(), a.end());
        bool flag;
        benchmark::DoNotOptimize(flag = (a.size() == size));
    }
}

BENCHMARK(BM_tbb_parallel_stable_sort)->Range(1 << min_array_size, 1 << max_array_size);