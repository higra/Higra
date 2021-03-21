/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "../test_utils.hpp"
#include "higra/algo/graph_core.hpp"
#include "higra/utils.hpp"
#include "higra/image/graph_image.hpp"
#include <set>

using namespace hg;

namespace test_graph_core {
    TEST_CASE("graph cut 2 labelisation", "[graph_algorithm]") {

        auto graph = get_4_adjacency_graph({3, 3});
        array_1d<char> edge_weights = {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0};

        auto labels = graph_cut_2_labelisation(graph, edge_weights);

        array_1d<index_t> ref_labels = {1, 2, 2, 1, 1, 3, 1, 3, 3};
        REQUIRE(is_in_bijection(labels, ref_labels));
    }

    TEST_CASE("labelisation 2 graph cut", "[graph_algorithm]") {
        auto graph = get_4_adjacency_graph({3, 3});
        array_1d<index_t> labels = {1, 2, 2, 1, 1, 3, 1, 3, 3};

        auto edge_weights = labelisation_2_graph_cut(graph, labels);

        array_1d<char> ref_edge_weights = {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0};

        REQUIRE(is_in_bijection(edge_weights, ref_edge_weights));
    }

    TEST_CASE("minimum spanning tree", "[graph_algorithm]") {
        auto graph = get_4_adjacency_graph({2, 3});

        xt::xarray<double> edge_weights{1, 0, 2, 1, 1, 1, 2};

        auto res = minimum_spanning_tree(graph, edge_weights);
        auto &mst = res.mst;
        auto &mst_edge_map = res.mst_edge_map;

        REQUIRE(num_vertices(mst) == 6);
        REQUIRE(num_edges(mst) == 5);
        std::vector<ugraph::edge_descriptor> ref = {{0, 3, 0},
                                                    {0, 1, 1},
                                                    {1, 4, 2},
                                                    {2, 5, 3},
                                                    {1, 2, 4}};
        for (index_t i = 0; i < (index_t) ref.size(); i++) {
            auto e = edge_from_index(i, mst);
            REQUIRE(e == ref[i]);
        }
        REQUIRE(xt::equal(mst_edge_map, array_1d<int>({1, 0, 3, 4, 2}))());
    }

    TEST_CASE("minimum spanning forest", "[graph_algorithm]") {
        ugraph graph(6);
        add_edge(0, 1, graph);
        add_edge(0, 2, graph);
        add_edge(1, 2, graph);

        add_edge(3, 4, graph);
        add_edge(3, 5, graph);
        add_edge(4, 5, graph);

        xt::xarray<double> edge_weights{0, 1, 2, 3, 4, 5};

        auto res = minimum_spanning_tree(graph, edge_weights);
        auto &mst = res.mst;
        auto &mst_edge_map = res.mst_edge_map;

        REQUIRE(num_vertices(mst) == 6);
        REQUIRE(num_edges(mst) == 4);
        std::vector<ugraph::edge_descriptor> ref = {{0, 1, 0},
                                                    {0, 2, 1},
                                                    {3, 4, 2},
                                                    {3, 5, 3}};
        for (index_t i = 0; i < (index_t) ref.size(); i++) {
            auto e = edge_from_index(i, mst);
            REQUIRE(e == ref[i]);
        }

        REQUIRE((mst_edge_map == array_1d<int>({0, 1, 3, 4})));
    }

    TEST_CASE("subgraph_spanning", "[graph_algorithm]") {
        auto graph = get_4_adjacency_graph({2, 2});
        array_1d<index_t> edge_indices = {3, 0};

        auto subgraph = subgraph_spanning(graph, edge_indices);
        REQUIRE(num_vertices(subgraph) == num_vertices(graph));
        REQUIRE(num_edges(subgraph) == edge_indices.size());
        std::vector<ugraph::edge_descriptor> ref = {{2, 3, 0},
                                                    {0, 1, 1}};
        for (index_t i = 0; i < (index_t) ref.size(); i++) {
            auto e = edge_from_index(i, subgraph);
            REQUIRE(e == ref[i]);
        }
    }

    TEST_CASE("line_graph trivial", "[graph_algorithm]") {
        ugraph graph(3);

        auto linegraph = line_graph(graph);
        REQUIRE(num_vertices(linegraph) == 0);
        REQUIRE(num_edges(linegraph) == 0);
    }

    TEST_CASE("line_graph trivial 2", "[graph_algorithm]") {
        ugraph graph(4);
        add_edge(0, 1, graph);
        add_edge(2, 3, graph);

        auto linegraph = line_graph(graph);
        REQUIRE(num_vertices(linegraph) == 2);
        REQUIRE(num_edges(linegraph) == 0);
    }

    TEST_CASE("line_graph trivial loop", "[graph_algorithm]") {
        using edge_t = ugraph::edge_descriptor;
        ugraph graph(3);
        add_edge(0, 0, graph);
        add_edge(0, 1, graph);
        add_edge(0, 2, graph);

        auto linegraph = line_graph(graph);
        REQUIRE(num_vertices(linegraph) == 3);
        REQUIRE(num_edges(linegraph) == 3);
        REQUIRE(edge_from_index(0, linegraph) == edge_t(0, 1, 0));
        REQUIRE(edge_from_index(1, linegraph) == edge_t(0, 2, 1));
        REQUIRE(edge_from_index(2, linegraph) == edge_t(1, 2, 2));
    }

    TEST_CASE("line_graph multiple edges", "[graph_algorithm]") {
        using edge_t = ugraph::edge_descriptor;
        ugraph graph(3);
        add_edge(0, 1, graph);
        add_edge(0, 1, graph);
        add_edge(1, 2, graph);

        auto linegraph = line_graph(graph);
        REQUIRE(num_vertices(linegraph) == 3);
        REQUIRE(num_edges(linegraph) == 3);
        REQUIRE(edge_from_index(0, linegraph) == edge_t(0, 1, 0));
        REQUIRE(edge_from_index(1, linegraph) == edge_t(0, 2, 1));
        REQUIRE(edge_from_index(2, linegraph) == edge_t(1, 2, 2));
    }

    TEST_CASE("line_graph multiple trivial loops", "[graph_algorithm]") {
        using edge_t = ugraph::edge_descriptor;
        ugraph graph(2);
        add_edge(0, 0, graph);
        add_edge(0, 0, graph);
        add_edge(0, 1, graph);

        auto linegraph = line_graph(graph);
        REQUIRE(num_vertices(linegraph) == 3);
        REQUIRE(num_edges(linegraph) == 3);
        REQUIRE(edge_from_index(0, linegraph) == edge_t(0, 1, 0));
        REQUIRE(edge_from_index(1, linegraph) == edge_t(0, 2, 1));
        REQUIRE(edge_from_index(2, linegraph) == edge_t(1, 2, 2));
    }

    TEST_CASE("line_graph 8 adj graph", "[graph_algorithm]") {
        using edge_t = ugraph::edge_descriptor;
        auto graph = get_8_adjacency_graph({2, 2});

        auto linegraph = line_graph(graph);
        REQUIRE(num_vertices(linegraph) == 6);
        REQUIRE(num_edges(linegraph) == 12);

        std::vector<std::set<index_t>> ref = {
                {1, 2, 3, 4},
                {0, 2, 3, 5},
                {0, 1, 4, 5},
                {0, 1, 4, 5},
                {0, 2, 3, 5},
                {1, 2, 3, 4}
        };
        for (auto v: vertex_iterator(linegraph)) {
            std::set<index_t> res;
            for (const auto &e : out_edge_iterator(v, linegraph)) {
                res.insert(e.target);
            }
            REQUIRE(res == ref[v]);
        }
    }

    TEST_CASE("line_graph tree", "[graph_algorithm]") {
        using edge_t = tree::edge_descriptor;
        auto t = tree(array_1d<index_t>{5, 5, 6, 6, 6, 7, 8, 8, 8});

        t.compute_children(); // required to process t as a graph

        auto linegraph = line_graph(t);
        REQUIRE(num_vertices(linegraph) == 8);
        REQUIRE(num_edges(linegraph) == 11);

        std::vector<std::set<index_t>> ref = {
                {1, 5},
                {0, 5},
                {3, 4, 6},
                {2, 4, 6},
                {2, 3, 6},
                {0, 1, 7},
                {2, 3, 4, 7},
                {5, 6},
        };
        for (auto v: vertex_iterator(linegraph)) {
            std::set<index_t> res;
            for (const auto &e : out_edge_iterator(v, linegraph)) {
                res.insert(e.target);
            }
            REQUIRE(res == ref[v]);
        }
    }
}
