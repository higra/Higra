/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/
#include <boost/test/unit_test.hpp>

#include "higra/graph.hpp"
#include "test_utils.hpp"

BOOST_AUTO_TEST_SUITE(regularGraph);

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

    BOOST_AUTO_TEST_CASE(sizeRegularGraph) {
        auto g = data.g;

        BOOST_CHECK(num_vertices(g) == 6);
    }

    BOOST_AUTO_TEST_CASE(vertexIteratorSimpleGraph) {

        auto g = data.g;

        vector<unsigned long> vref{0, 1, 2, 3, 4, 5};
        vector<unsigned long> vtest;

        for (auto v: hg::vertex_iterator(g)) {
            vtest.push_back(v);
        }

        BOOST_CHECK(vectorEqual(vref, vtest));

    }

    BOOST_AUTO_TEST_CASE(outEdgeIteratorNeighbouringGraph) {

        auto g = data.g;

        vector<vector<pair<unsigned long, unsigned long>>> outListsRef{{{0, 1}, {0, 3}},
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
            BOOST_CHECK(vectorEqual(outListsRef[v], outListsTest[v]));
            BOOST_CHECK(out_degree(v, g) == outListsRef[v].size());

        }

    }

    BOOST_AUTO_TEST_CASE(inEdgeIteratorSimpleGraph) {

        auto g = data.g;

        vector<vector<pair<unsigned long, unsigned long>>> inListsRef{{{1, 0}, {3, 0}},
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
            BOOST_CHECK(vectorEqual(inListsRef[v], inListsTest[v]));
            BOOST_CHECK(in_degree(v, g) == inListsRef[v].size());
            BOOST_CHECK(degree(v, g) == inListsRef[v].size());
        }
    }


    BOOST_AUTO_TEST_CASE(adjacentVertexIteratorSimpleGraph) {

        auto g = data.g;

        vector<vector<unsigned long>> adjListsRef{{1, 3},
                                                  {0, 2, 4},
                                                  {1, 5},
                                                  {0, 4},
                                                  {1, 3, 5},
                                                  {2, 4}};
        vector<vector<unsigned long>> adjListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            adjListsTest.push_back(vector<unsigned long>());
            for (auto av: hg::adjacent_vertex_iterator(v, g)) {
                adjListsTest[v].push_back(av);
            }
            BOOST_CHECK(vectorEqual(adjListsRef[v], adjListsTest[v]));
        }
    }

BOOST_AUTO_TEST_SUITE_END();