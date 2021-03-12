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

#include "higra/image/graph_image.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "higra/hierarchy/watershed_hierarchy.hpp"
#include "xtensor/xrandom.hpp"
#include "higra/structure/lca_fast.hpp"

using namespace xt;
using namespace hg;

index_t repetition = 1;

static void BM_lca_sparse_table_block(benchmark::State &state) {
    for (auto _ : state) {
        for(index_t i = 0; i < repetition; i++){
            state.PauseTiming();

            index_t size = state.range(0);
            index_t bsize = state.range(1);


            auto g = get_4_adjacency_graph({size, size});
            array_1d<double> weights = xt::random::rand<double>({num_edges(g)});
            auto res = watershed_hierarchy_by_area(g, weights);
            auto & tree = res.tree;

            state.ResumeTiming();
            lca_sparse_table_block l(tree, bsize);
            auto ll = l.lca(sources(g), targets(g));

            benchmark::DoNotOptimize(ll[0]);
        }
    }
}


static void gridSearch(benchmark::internal::Benchmark* b) {
    for (index_t i = 256; i <= 2048; i+=256)
        for (int j = 32; j <= 4096; j *= 2)
            b->Args({i, j});
}


BENCHMARK(BM_lca_sparse_table_block)->Apply(gridSearch);



static void BM_lca_sparse_table(benchmark::State &state) {
    for (auto _ : state) {
        for(index_t i = 0; i < repetition; i++){
            state.PauseTiming();

            index_t size = state.range(0);

            auto g = get_4_adjacency_graph({size, size});
            array_1d<double> weights = xt::random::rand<double>({num_edges(g)});
            auto res = watershed_hierarchy_by_area(g, weights);
            auto & tree = res.tree;

            state.ResumeTiming();
            lca_sparse_table l(tree);
            auto ll = l.lca(sources(g), targets(g));

            benchmark::DoNotOptimize(ll[0]);
        }
    }
}

BENCHMARK(BM_lca_sparse_table)->DenseRange(256, 2048, 256);