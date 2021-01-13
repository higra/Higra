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

#include <iostream>
#include "xtensor/xio.hpp"
using namespace xt;
using namespace hg;


std::size_t min_size = 6;
std::size_t max_size = 12;

static void BM_from_edge_list_no_preallocation(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        hg::index_t size = state.range(0);
        xt::random::seed(42);

        array_2d<index_t> vertices = reshape_view(arange<index_t>({size * size}), {size, size});
        array_1d<index_t> h_edges_sources = flatten(view(vertices, all(), range(0, size - 1)));
        array_1d<index_t> h_edges_targets = flatten(view(vertices, all(), range(1, size)));

        array_1d<index_t> v_edges_sources = flatten(view(vertices, range(0, size - 1), all()));
        array_1d<index_t> v_edges_targets = flatten(view(vertices, range(1, size), all()));

        array_1d<index_t> edges_sources = concatenate(xtuple(h_edges_sources, v_edges_sources));
        array_1d<index_t> edges_targets = concatenate(xtuple(h_edges_targets, v_edges_targets));

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
        xt::random::seed(42);

        array_2d<index_t> vertices = reshape_view(arange<index_t>({size * size}), {size, size});

        array_1d<index_t> h_edges_sources = flatten(view(vertices, all(), range(0, size - 1)));
        array_1d<index_t> h_edges_targets = flatten(view(vertices, all(), range(1, size)));

        array_1d<index_t> v_edges_sources = flatten(view(vertices, range(0, size - 1), all()));
        array_1d<index_t> v_edges_targets = flatten(view(vertices, range(1, size), all()));

        array_1d<index_t> edges_sources = concatenate(xtuple(h_edges_sources, v_edges_sources));
        array_1d<index_t> edges_targets = concatenate(xtuple(h_edges_targets, v_edges_targets));

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
        xt::random::seed(42);

        state.ResumeTiming();
        auto res = hg::get_4_adjacency_graph(hg::embedding_grid_2d({size, size}));

        benchmark::DoNotOptimize(res.num_vertices());
    }
}

BENCHMARK(BM_graph_implicit_to_explicit)->Range(1 << min_size, 1 << max_size);

