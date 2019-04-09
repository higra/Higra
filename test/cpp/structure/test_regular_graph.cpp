/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/graph.hpp"
#include "../test_utils.hpp"

namespace regular_graph {

    using namespace hg;
    using namespace std;

    struct _data {

        hg::regular_grid_graph_2d g;

        _data() {
            hg::embedding_grid_2d embedding{2, 3}; // 2 rows, 3 columns
            std::vector<point_2d_i> neighbours{{{-1, 0}},
                                               {{0,  -1}},
                                               {{0,  1}},
                                               {{1,  0}}}; // 4 adjacency

            g = hg::regular_grid_graph_2d(embedding, neighbours);
        }

    } data;

    TEST_CASE("regular graph size", "[regular_graph]") {
        auto g = data.g;

        REQUIRE(num_vertices(g) == 6);
    }

    TEST_CASE("vertex iterator", "[regular_graph]") {
        auto g = data.g;

        vector<unsigned long> vref{0, 1, 2, 3, 4, 5};
        vector<unsigned long> vtest;

        for (auto v: hg::vertex_iterator(g)) {
            vtest.push_back(v);
        }

        REQUIRE(vectorEqual(vref, vtest));
    }

    TEST_CASE("regular graph out edge iterator", "[regular_graph]") {
        auto g = data.g;

        vector<vector<pair<unsigned long, unsigned long>>> outListsRef{
                {{0, 1}, {0, 3}},
                {{1, 0}, {1, 2}, {1, 4}},
                {{2, 1}, {2, 5}},
                {{3, 0}, {3, 4}},
                {{4, 1}, {4, 3}, {4, 5}},
                {{5, 2}, {5, 4}}
        };
        vector<vector<pair<unsigned long, unsigned long>>> outListsTest;

        for (size_t v = 0; v < 6; v++) {
            outListsTest.push_back({});
            for (auto e: hg::out_edge_iterator(v, g))
                outListsTest[v].push_back({source(e, g), target(e, g)});
            REQUIRE(vectorEqual(outListsRef[v], outListsTest[v]));
            REQUIRE(out_degree(v, g) == outListsRef[v].size());
        }
    }

    TEST_CASE("regular graph in edge iterator", "[regular_graph]") {
        auto g = data.g;

        vector<vector<pair<unsigned long, unsigned long>>> inListsRef{
                {{1, 0}, {3, 0}},
                {{0, 1}, {2, 1}, {4, 1}},
                {{1, 2}, {5, 2}},
                {{0, 3}, {4, 3}},
                {{1, 4}, {3, 4}, {5, 4}},
                {{2, 5}, {4, 5}}
        };
        vector<vector<pair<unsigned long, unsigned long>>> inListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            inListsTest.push_back(vector<pair<unsigned long, unsigned long>>());
            for (auto e: hg::in_edge_iterator(v, g))
                inListsTest[v].push_back({source(e, g), target(e, g)});
            REQUIRE(vectorEqual(inListsRef[v], inListsTest[v]));
            REQUIRE(in_degree(v, g) == inListsRef[v].size());
            REQUIRE(degree(v, g) == inListsRef[v].size());
        }
    }


    TEST_CASE("regular graph adjacent vertex iterator", "[regular_graph]") {
        auto g = data.g;

        vector<vector<unsigned long>> adjListsRef{
                {1, 3},
                {0, 2, 4},
                {1, 5},
                {0, 4},
                {1, 3, 5},
                {2, 4}
        };
        vector<vector<unsigned long>> adjListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            adjListsTest.push_back(vector<unsigned long>());
            for (auto av: hg::adjacent_vertex_iterator(v, g)) {
                adjListsTest[v].push_back(av);
            }
            REQUIRE(vectorEqual(adjListsRef[v], adjListsTest[v]));
        }
    }
}