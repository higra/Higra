/***************************************************************************
* Copyright ESIEE Paris (2023)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/algo/bipartite_graph.hpp"
#include "../test_utils.hpp"
#include <set>
#include "xtensor/generators/xrandom.hpp"

using namespace hg;

namespace test_alignment {

    TEST_CASE("bipartite graph matching", "[bipartite_graph]") {

        ugraph g(6);
        add_edges(array_1d<index_t>{0, 0, 1, 1, 2}, array_1d<index_t>{3, 4, 3, 5, 5}, g);
        array_1d<index_t> weights{1, 1, 1, 1, 1};

        auto edges = bipartite_graph_matching(g, weights);

        std::set<index_t> edge_set{4, 2, 1};

        size_t n = num_vertices(g) / 2;
        REQUIRE(edges.size() == n);
        for (index_t i = 0; i < (index_t) n; ++i) {
            REQUIRE(edge_set.count(edges[i]) == 1);
            edge_set.erase(edges[i]);
        }
    }

    TEST_CASE("bipartite graph matching weighted", "[bipartite_graph]") {

        ugraph g(6);
        add_edges(array_1d<index_t>{0, 0, 1, 1, 2, 1}, array_1d<index_t>{3, 4, 3, 5, 5, 4}, g);
        array_1d<index_t> weights{3, 1, 6, 0, 10, 2};

        auto edges = bipartite_graph_matching(g, weights);

        std::set<index_t> edge_set{4, 0, 5};

        size_t n = num_vertices(g) / 2;
        REQUIRE(edges.size() == n);
        for (index_t i = 0; i < (index_t) n; ++i) {
            REQUIRE(edge_set.count(edges[i]) == 1);
            edge_set.erase(edges[i]);
        }
    }

    TEST_CASE("is bipartite graph depth first empty", "[bipartite_graph]") {
        ugraph g(0);

        auto res = is_bipartite_graph(g);
        auto &ans = res.first;
        auto &color = res.second;

        REQUIRE(ans);
        REQUIRE(color.size() == 0);
    }

    TEST_CASE("is bipartite graph union find empty", "[bipartite_graph]") {
        ugraph g(0);

        auto res = is_bipartite_graph(g.sources(), g.targets(), g.num_vertices());
        auto &ans = res.first;
        auto &color = res.second;

        REQUIRE(ans);
        REQUIRE(color.size() == 0);
    }

    TEST_CASE("is bipartite graph depth first search", "[bipartite_graph]") {
        ugraph g(6);
        add_edges(array_1d<index_t>{0, 0, 1, 1, 2, 1}, array_1d<index_t>{3, 4, 3, 5, 5, 4}, g);

        auto res = is_bipartite_graph(g);
        auto &ans = res.first;
        auto &color = res.second;

        REQUIRE(ans);
        REQUIRE(color.size() == 6);
        for (const auto & e: edge_iterator(g)){
            REQUIRE(color[e.source] != color[e.target]);
        }
    }


    TEST_CASE("is bipartite graph union find search", "[bipartite_graph]") {
        ugraph g(6);
        add_edges(array_1d<index_t>{0, 0, 1, 1, 2, 1}, array_1d<index_t>{3, 4, 3, 5, 5, 4}, g);

        auto res = is_bipartite_graph(g.sources(), g.targets(), g.num_vertices());
        auto &ans = res.first;
        auto &color = res.second;

        REQUIRE(ans);
        REQUIRE(color.size() == 6);
        for (const auto & e: edge_iterator(g)){
            REQUIRE(color[e.source] != color[e.target]);
        }
    }

    TEST_CASE("is bipartite graph depth first search mixed", "[bipartite_graph]") {
        ugraph g(6);
        add_edges(array_1d<index_t>{0, 0, 4, 2}, array_1d<index_t>{1, 4, 3, 3}, g);

        auto res = is_bipartite_graph(g);
        auto &ans = res.first;
        auto &color = res.second;

        REQUIRE(ans);
        REQUIRE(color.size() == 6);
        for (const auto &e: edge_iterator(g)){
            REQUIRE(color[e.source] != color[e.target]);
        }
    }

    TEST_CASE("is bipartite graph union find search mixed", "[bipartite_graph]") {
        ugraph g(6);
        add_edges(array_1d<index_t>{0, 0, 4, 2}, array_1d<index_t>{1, 4, 3, 3}, g);

        auto res = is_bipartite_graph(g.sources(), g.targets(), g.num_vertices());
        auto &ans = res.first;
        auto &color = res.second;

        REQUIRE(ans);
        REQUIRE(color.size() == 6);
        for (const auto &e: edge_iterator(g)){
            REQUIRE(color[e.source] != color[e.target]);
        }
    }

    TEST_CASE("is bipartite graph depth first search false", "[bipartite_graph]") {
        ugraph g(6);
        add_edges(array_1d<index_t>{0, 0, 1, 1, 2, 1, 5}, array_1d<index_t>{3, 4, 3, 5, 5, 4, 4}, g);

        auto res = is_bipartite_graph(g);
        auto &ans = res.first;
        auto &color = res.second;

        REQUIRE(!ans);
        REQUIRE(color.size() == 0);
    }

    TEST_CASE("is bipartite graph union find search false", "[bipartite_graph]") {
        ugraph g(6);
        add_edges(array_1d<index_t>{0, 0, 1, 1, 2, 1, 5}, array_1d<index_t>{3, 4, 3, 5, 5, 4, 4}, g);

        auto res = is_bipartite_graph(g.sources(), g.targets(), g.num_vertices());
        auto &ans = res.first;
        auto &color = res.second;

        REQUIRE(!ans);
        REQUIRE(color.size() == 0);
    }

    TEST_CASE("is bipartite graph depth first search randomized", "[bipartite_graph]") {
        // create random bipartite graph of size 100
        index_t n = 100;
        index_t m = 300;
        ugraph g(n);
        index_t split = n * 4 / 5;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<index_t> dis(0, split - 1);
        std::uniform_int_distribution<index_t> dis2(split, n - 1);


        array_1d<index_t> randomized_order = xt::random::permutation(n);
        for (index_t i = 0; i < m; ++i) {
            add_edge(randomized_order(dis(gen)), randomized_order(dis2(gen)), g);
        }


        auto res = is_bipartite_graph(g);
        auto &ans = res.first;
        auto &color = res.second;

        REQUIRE(ans);
        REQUIRE(color.size() == (size_t)n);
        for (const auto &e: edge_iterator(g)){
            REQUIRE(color[e.source] != color[e.target]);
        }
    }

    TEST_CASE("is bipartite graph union find randomized", "[bipartite_graph]") {
        // create random bipartite graph of size 100
        index_t n = 100;
        index_t m = 300;
        ugraph g(n);
        index_t split = n * 4 / 5;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<index_t> dis(0, split - 1);
        std::uniform_int_distribution<index_t> dis2(split, n - 1);


        array_1d<index_t> randomized_order = xt::random::permutation(n);
        for (index_t i = 0; i < m; ++i) {
            add_edge(randomized_order(dis(gen)), randomized_order(dis2(gen)), g);
        }


        auto res = is_bipartite_graph(g.sources(), g.targets(), g.num_vertices());
        auto &ans = res.first;
        auto &color = res.second;

        REQUIRE(ans);
        REQUIRE(color.size() == (size_t)n);
        for (const auto &e: edge_iterator(g)){
            REQUIRE(color[e.source] != color[e.target]);
        }
    }


}