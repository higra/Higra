//
// Created by user on 3/9/18.
//

#include <boost/test/unit_test.hpp>

#include "higra/graph.hpp"
#include "xtensor/xio.hpp"

#include "test_utils.hpp"


/**
 * graph tests
 */


BOOST_AUTO_TEST_SUITE(undirectedGraph);


    using namespace std;
    using namespace hg;


    struct _data {
        // 0 - 1
        // | /
        // 2   3
        hg::ugraph g;

        _data() : g(4ul) {
            add_edge(0, 1, g);
            add_edge(1, 2, g);
            add_edge(0, 2, g);
        }

    } data;

    BOOST_AUTO_TEST_CASE(sizeSimpleGraph) {
        auto g = data.g;

        BOOST_CHECK(num_vertices(g) == 4);
        BOOST_CHECK(num_edges(g) == 3);
        BOOST_CHECK(out_degree(0, g) == 2);
        BOOST_CHECK(in_degree(0, g) == 2);
        BOOST_CHECK(degree(0, g) == 2);
        BOOST_CHECK(out_degree(3, g) == 0);
        BOOST_CHECK(in_degree(3, g) == 0);
        BOOST_CHECK(degree(3, g) == 0);
    }

    BOOST_AUTO_TEST_CASE(copyCTR) {
        auto g0 = data.g;
        auto g = hg::copy_graph(g0);

        vector<pair<ulong, ulong>> eref{{0, 1},
                                        {1, 2},
                                        {0, 2}};
        vector<pair<ulong, ulong>> etest;
        for (auto e: hg::edge_iterator(g)) {
            etest.push_back({source(e, g), target(e, g)});
        }
        BOOST_CHECK(vectorEqual(eref, etest));

    }

    BOOST_AUTO_TEST_CASE(copyCTR2) {
        hg::embedding_grid_2d embedding{2, 3}; // 2 rows, 3 columns
        std::vector<point_2d_i> neighbours{{      -1, 0},
                                                 {0,  -1},
                                                 {0,  1},
                                                 {1,  0}}; // 4 adjacency

        auto g0 = hg::regular_grid_graph_2d(embedding, neighbours);
        auto g = hg::copy_graph(g0);

        vector<vector<pair<ulong, ulong>>> outListsRef{{{0, 1}, {0, 3}},
                                                       {{1, 0}, {1, 2}, {1, 4}},
                                                       {{2, 1}, {2, 5}},
                                                       {{3, 0}, {3, 4}},
                                                       {{4, 1}, {4, 3}, {4, 5}},
                                                       {{5, 2}, {5, 4}}
        };
        vector<vector<pair<ulong, ulong>>> outListsTest;

        for (size_t v = 0; v < 6; v++) {
            outListsTest.push_back({});
            for (auto e: hg::out_edge_iterator(v, g))
                outListsTest[v].push_back({source(e, g), target(e, g)});
            BOOST_CHECK(vectorEqual(outListsRef[v], outListsTest[v]));
            BOOST_CHECK(out_degree(v, g) == outListsRef[v].size());

        }


    }


    BOOST_AUTO_TEST_CASE(vertexIteratorSimpleGraph) {

        auto g = data.g;

        vector<ulong> vref{0, 1, 2, 3};
        vector<ulong> vtest;

        for (auto v: hg::vertex_iterator(g))
            vtest.push_back(v);

        BOOST_CHECK(vectorEqual(vref, vtest));

    }

    BOOST_AUTO_TEST_CASE(edgeIteratorSimpleGraph) {

        auto g = data.g;

        vector<pair<ulong, ulong>> eref{{0, 1},
                                        {1, 2},
                                        {0, 2}};
        vector<pair<ulong, ulong>> etest;
        for (auto e: hg::edge_iterator(g)) {
            etest.push_back({source(e, g), target(e, g)});
        }

        BOOST_CHECK(vectorEqual(eref, etest));
    }

    BOOST_AUTO_TEST_CASE(outEdgeIteratorSimpleGraph) {

        auto g = data.g;

        vector<vector<pair<ulong, ulong>>> outListsRef{{{0, 1}, {0, 2}},
                                                       {{1, 0}, {1, 2}},
                                                       {{2, 1}, {2, 0}},
                                                       {}};
        vector<vector<pair<ulong, ulong>>> outListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            outListsTest.push_back(vector<pair<ulong, ulong>>());
            for (auto e: hg::out_edge_iterator(v, g)) {
                //showTypeName<decltype(e)>("edge type : ");
                outListsTest[v].push_back({source(e, g), target(e, g)});
            }
            BOOST_CHECK(vectorEqual(outListsRef[v], outListsTest[v]));
        }

    }

    BOOST_AUTO_TEST_CASE(inEdgeIteratorSimpleGraph) {

        auto g = data.g;

        vector<vector<pair<ulong, ulong>>> inListsRef{{{1, 0}, {2, 0}},
                                                      {{0, 1}, {2, 1}},
                                                      {{1, 2}, {0, 2}},
                                                      {}};
        vector<vector<pair<ulong, ulong>>> inListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            inListsTest.push_back(vector<pair<ulong, ulong>>());
            for (auto e: hg::in_edge_iterator(v, g))
                inListsTest[v].push_back({source(e, g), target(e, g)});
            BOOST_CHECK(vectorEqual(inListsRef[v], inListsTest[v]));
        }
    }

    BOOST_AUTO_TEST_CASE(adjacentVertexIteratorSimpleGraph) {

        auto g = data.g;

        vector<vector<ulong>> adjListsRef{{1, 2},
                                          {0, 2},
                                          {1, 0},
                                          {}};
        vector<vector<ulong>> adjListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            adjListsTest.push_back(vector<ulong>());
            for (auto av: hg::adjacent_vertex_iterator(v, g)) {
                adjListsTest[v].push_back(av);
            }
            BOOST_CHECK(vectorEqual(adjListsRef[v], adjListsTest[v]));
        }
    }

    BOOST_AUTO_TEST_CASE(edgeIndexIteratorSimpleGraph) {

        auto g = data.g;

        vector<ulong> ref{0, 1, 2};
        vector<ulong> test;

        for (auto v: hg::edge_index_iterator(g)) {
            test.push_back(v);
        }
        BOOST_CHECK(vectorEqual(ref, test));
    }

    BOOST_AUTO_TEST_CASE(outEdgeIndexIteratorSimpleGraph) {

        auto g = data.g;

        vector<vector<ulong>> ref{{0, 2},
                                  {0, 1},
                                  {1, 2},
                                  {}};
        vector<vector<ulong>> test;

        for (auto v: hg::vertex_iterator(g)) {
            test.push_back(vector<ulong>());
            for (auto av: hg::out_edge_index_iterator(v, g)) {
                test[v].push_back(av);
            }
            BOOST_CHECK(vectorEqual(ref[v], test[v]));
        }
    }

    BOOST_AUTO_TEST_CASE(inEdgeIndexIteratorSimpleGraph) {

        auto g = data.g;

        vector<vector<ulong>> ref{{0, 2},
                                  {0, 1},
                                  {1, 2},
                                  {}};
        vector<vector<ulong>> test;

        for (auto v: hg::vertex_iterator(g)) {
            test.push_back(vector<ulong>());
            for (auto av: hg::in_edge_index_iterator(v, g)) {
                test[v].push_back(av);
            }
            BOOST_CHECK(vectorEqual(ref[v], test[v]));
        }
    }

BOOST_AUTO_TEST_SUITE_END();