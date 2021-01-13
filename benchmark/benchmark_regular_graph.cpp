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
#include "higra/graph.hpp"
#include "xtensor/xrandom.hpp"
#include "higra/image/graph_image.hpp"

using namespace xt;
using namespace hg;


std::size_t min_size = 6;
std::size_t max_size = 12;


static void BM_graph_implicit_adjacency_iterator(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        hg::index_t size = state.range(0);
        xt::random::seed(42);
        auto g = hg::get_4_adjacency_implicit_graph(hg::embedding_grid_2d({size, size}));

        state.ResumeTiming();
        index_t sum = 0;
        for(auto v: hg::vertex_iterator(g)){
            for(auto n: hg::adjacent_vertex_iterator(v, g)){
                sum += n;
            }
        }

        benchmark::DoNotOptimize(++sum);
    }
}

BENCHMARK(BM_graph_implicit_adjacency_iterator)->Range(1 << min_size, 1 << max_size);


