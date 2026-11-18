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

#include "higra/image/graph_image.hpp"

#include <iostream>
using namespace hg;


std::size_t min_size = 6;
std::size_t max_size = 12;

static void BM_from_edge_list_no_preallocation(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        hg::index_t size = state.range(0);
        bench_utils::random_seed(42);

        auto edges = bench_utils::grid_4adj_edges(size);
        const auto &edges_sources = edges.first;
        const auto &edges_targets = edges.second;

        state.ResumeTiming();
        ugraph g(size * size);
        add_edges(edges_sources, edges_targets, g);

        benchmark::DoNotOptimize(g.num_vertices());
    }
}

BENCHMARK(BM_from_edge_list_no_preallocation)->Range(1 << min_size, 1 << max_size);

static void BM_from_edge_list_with_preallocation(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        hg::index_t size = state.range(0);
        bench_utils::random_seed(42);

        auto edges = bench_utils::grid_4adj_edges(size);
        const auto &edges_sources = edges.first;
        const auto &edges_targets = edges.second;

        state.ResumeTiming();
        ugraph g(size * size, edges_sources.size(), 4);
        add_edges(edges_sources, edges_targets, g);

        benchmark::DoNotOptimize(g.num_vertices());
    }
}

BENCHMARK(BM_from_edge_list_with_preallocation)->Range(1 << min_size, 1 << max_size);

static void BM_graph_implicit_to_explicit(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        hg::index_t size = state.range(0);
        bench_utils::random_seed(42);

        state.ResumeTiming();
        auto res = hg::get_4_adjacency_graph(hg::embedding_grid_2d({size, size}));

        benchmark::DoNotOptimize(res.num_vertices());
    }
}

BENCHMARK(BM_graph_implicit_to_explicit)->Range(1 << min_size, 1 << max_size);

