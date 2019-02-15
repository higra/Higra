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
#include <boost/mpl/list.hpp>
#include "higra/graph.hpp"
#include "../test_utils.hpp"


/**
 * graph tests
 */

typedef boost::mpl::list<hg::ugraph, hg::undirected_graph<hg::hash_setS> > test_types;

BOOST_AUTO_TEST_SUITE(undirectedGraph);




    using namespace std;
    using namespace hg;

    template<typename T>
    struct data {
        // 0 - 1
        // | /
        // 2   3
        static auto g() {
            T g(4ul);
            add_edge(0, 1, g);
            add_edge(1, 2, g);
            add_edge(0, 2, g);
            return g;
        }

    };

    BOOST_AUTO_TEST_CASE_TEMPLATE(sizeSimpleGraph, T, test_types) {
        auto g = data<T>::g();

        BOOST_CHECK(num_vertices(g) == 4);
        BOOST_CHECK(num_edges(g) == 3);
        BOOST_CHECK(out_degree(0, g) == 2);
        BOOST_CHECK(in_degree(0, g) == 2);
        BOOST_CHECK(degree(0, g) == 2);
        BOOST_CHECK(out_degree(3, g) == 0);
        BOOST_CHECK(in_degree(3, g) == 0);
        BOOST_CHECK(degree(3, g) == 0);

        array_2d<index_t> indices{{0, 3},
                                  {1, 2}};
        array_2d<size_t> ref{{2, 0},
                             {2, 2}};

        BOOST_CHECK(xt::allclose(degree(indices, g), ref));
        BOOST_CHECK(xt::allclose(in_degree(indices, g), ref));
        BOOST_CHECK(xt::allclose(out_degree(indices, g), ref));
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(copyCTR, T, test_types) {
        auto g0 = data<T>::g();
        auto g = hg::copy_graph<T>(g0);

        vector<pair<index_t, index_t>> eref{{0, 1},
                                            {1, 2},
                                            {0, 2}};
        vector<pair<index_t, index_t>> etest;
        for (auto e: hg::edge_iterator(g)) {
            etest.push_back(e);
        }

        BOOST_CHECK(vectorSame(eref, etest));

    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(copyCTR2, T, test_types) {
        hg::embedding_grid_2d embedding{2, 3}; // 2 rows, 3 columns
        std::vector<point_2d_i> neighbours{point_2d_i{{-1l, 0l}},
                                           point_2d_i{{0l, -1l}},
                                           point_2d_i{{0l, 1l}},
                                           point_2d_i{{1l, 0l}}}; // 4 adjacency

        auto g0 = hg::regular_grid_graph_2d(embedding, neighbours);
        auto g = hg::copy_graph<T>(g0);

        vector<vector<pair<index_t, index_t>>> outListsRef{{{0, 1}, {0, 3}},
                                                           {{1, 0}, {1, 2}, {1, 4}},
                                                           {{2, 1}, {2, 5}},
                                                           {{3, 0}, {3, 4}},
                                                           {{4, 1}, {4, 3}, {4, 5}},
                                                           {{5, 2}, {5, 4}}
        };
        vector<vector<pair<index_t, index_t>>> outListsTest;

        for (size_t v = 0; v < 6; v++) {
            outListsTest.push_back({});
            for (auto e: hg::out_edge_iterator(v, g))
                outListsTest[v].push_back({source(e, g), target(e, g)});
            BOOST_CHECK(vectorSame(outListsRef[v], outListsTest[v]));
            BOOST_CHECK(out_degree(v, g) == outListsRef[v].size());

        }
    }


    BOOST_AUTO_TEST_CASE_TEMPLATE(vertexIteratorSimpleGraph, T, test_types) {

        auto g = data<T>::g();

        vector<index_t> vref{0, 1, 2, 3};
        vector<index_t> vtest;

        for (auto v: hg::vertex_iterator(g))
            vtest.push_back(v);

        BOOST_CHECK(vectorEqual(vref, vtest));

    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(edgeIteratorSimpleGraph, T, test_types) {

        auto g = data<T>::g();

        vector<pair<index_t, index_t>> eref{{0, 1},
                                            {1, 2},
                                            {0, 2}};
        vector<pair<index_t, index_t>> etest;
        for (auto e: hg::edge_iterator(g)) {
            etest.push_back(e);
        }

        BOOST_CHECK(vectorEqual(eref, etest));
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(outEdgeIteratorSimpleGraph, T, test_types) {

        auto g = data<T>::g();

        vector<vector<pair<index_t, index_t>>> outListsRef{{{0, 1}, {0, 2}},
                                                           {{1, 0}, {1, 2}},
                                                           {{2, 1}, {2, 0}},
                                                           {}};
        vector<vector<pair<index_t, index_t>>> outListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            outListsTest.push_back(vector<pair<index_t, index_t>>());
            for (auto e: hg::out_edge_iterator(v, g)) {
                outListsTest[v].push_back({source(e, g), target(e, g)});
            }
            BOOST_CHECK(vectorSame(outListsRef[v], outListsTest[v]));
        }

    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(inEdgeIteratorSimpleGraph, T, test_types) {

        auto g = data<T>::g();

        vector<vector<pair<index_t, index_t>>> inListsRef{{{1, 0}, {2, 0}},
                                                          {{0, 1}, {2, 1}},
                                                          {{1, 2}, {0, 2}},
                                                          {}};

        for (auto v: hg::vertex_iterator(g)) {
            vector<pair<index_t, index_t>> inListTest;
            for (auto e: hg::in_edge_iterator(v, g))
                inListTest.push_back(e);
            BOOST_CHECK(vectorSame(inListsRef[v], inListTest));
        }
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(addEdges, T, test_types) {
        auto g = data<T>::g();

        ugraph g2(4);

        array_1d<int> sources{0, 1, 0};
        array_1d<int> targets{1, 2, 2};
        add_edges(sources, targets, g2);

        BOOST_CHECK(num_edges(g2) == 3);

        for (index_t i = 0; i < num_edges(g2); i++) {
            auto e1 = edge_from_index(i, g);
            auto e2 = edge_from_index(i, g2);
            BOOST_CHECK(e1 == e2);
        }
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(adjacentVertexIteratorSimpleGraph, T, test_types) {

        auto g = data<T>::g();

        vector<vector<index_t>> adjListsRef{{1, 2},
                                            {0, 2},
                                            {1, 0},
                                            {}};
        vector<vector<index_t>> adjListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            adjListsTest.push_back(vector<index_t>());
            for (auto av: hg::adjacent_vertex_iterator(v, g)) {
                adjListsTest[v].push_back(av);
            }
            BOOST_CHECK(vectorSame(adjListsRef[v], adjListsTest[v]));
        }
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(edgeIndexIteratorSimpleGraph, T, test_types) {

        auto g = data<T>::g();

        vector<index_t> ref{0, 1, 2};
        vector<index_t> test;

        for (auto v: hg::edge_iterator(g)) {
            test.push_back(index(v, g));
        }
        BOOST_CHECK(vectorSame(ref, test));
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(outEdgeIndexIteratorSimpleGraph, T, test_types) {

        auto g = data<T>::g();

        vector<vector<index_t>> ref{{0, 2},
                                    {0, 1},
                                    {1, 2},
                                    {}};
        vector<vector<index_t>> test;

        for (auto v: hg::vertex_iterator(g)) {
            test.push_back(vector<index_t>());
            for (auto av: hg::out_edge_iterator(v, g)) {
                test[v].push_back(index(av, g));
            }
            BOOST_CHECK(vectorSame(ref[v], test[v]));
        }
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(inEdgeIndexIteratorSimpleGraph, T, test_types) {

        auto g = data<T>::g();

        vector<vector<index_t>> ref{{0, 2},
                                    {0, 1},
                                    {1, 2},
                                    {}};
        vector<vector<index_t>> test;

        for (auto v: hg::vertex_iterator(g)) {
            test.push_back(vector<index_t>());
            for (auto av: hg::in_edge_iterator(v, g)) {
                test[v].push_back(index(av, g));
            }
            BOOST_CHECK(vectorSame(ref[v], test[v]));
        }
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(edgeIndex, T, test_types) {

        auto g = data<T>::g();

        vector<pair<index_t, index_t>> eref{{0, 1},
                                            {1, 2},
                                            {0, 2}};
        vector<pair<index_t, index_t>> etest;
        for (auto e: hg::edge_iterator(g)) {
            etest.push_back(edge_from_index(index(e, g), g));
        }

        BOOST_CHECK(vectorSame(eref, etest));
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(removeEdge, T, test_types) {

        auto g = data<T>::g();

        remove_edge(1, g);

        vector<pair<index_t, index_t>> eref{{0,             1}, // deleted {1,2}
                                            {invalid_index, invalid_index},
                                            {0,             2}};
        vector<pair<index_t, index_t>> etest;
        for (auto e: hg::edge_iterator(g)) {
            etest.push_back(edge_from_index(e, g));
        }

        BOOST_CHECK(vectorSame(eref, etest));

        BOOST_CHECK(degree(0, g) == 2);
        BOOST_CHECK(degree(1, g) == 1);
        BOOST_CHECK(degree(2, g) == 1);

        vector<vector<index_t>> adjListsRef{{1, 2},
                                            {0},
                                            {0},
                                            {}};
        vector<vector<index_t>> adjListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            adjListsTest.push_back(vector<index_t>());
            for (auto av: hg::adjacent_vertex_iterator(v, g)) {
                adjListsTest[v].push_back(av);
            }
            BOOST_CHECK(vectorSame(adjListsRef[v], adjListsTest[v]));
        }
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(setEdge, T, test_types) {

        auto g = data<T>::g();

        set_edge(1, 3, 0, g);

        vector<pair<index_t, index_t>> eref{{0, 1}, // deleted {1,2}
                                            {0, 3},
                                            {0, 2}};
        vector<pair<index_t, index_t>> etest;
        for (auto e: hg::edge_iterator(g)) {
            etest.push_back(e);
        }

        BOOST_CHECK(vectorSame(eref, etest));

        BOOST_CHECK(degree(0, g) == 3);
        BOOST_CHECK(degree(1, g) == 1);
        BOOST_CHECK(degree(2, g) == 1);
        BOOST_CHECK(degree(3, g) == 1);

        vector<vector<index_t>> adjListsRef{{1, 2, 3},
                                            {0},
                                            {0},
                                            {0}};
        vector<vector<index_t>> adjListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            adjListsTest.push_back(vector<index_t>());
            for (auto av: hg::adjacent_vertex_iterator(v, g)) {
                adjListsTest[v].push_back(av);
            }
            BOOST_CHECK(vectorSame(adjListsRef[v], adjListsTest[v]));
        }
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(adjacency_matrix, T, test_types) {

        T g(5);
        add_edge(0, 1, g);
        add_edge(0, 2, g);
        add_edge(0, 3, g);
        add_edge(0, 4, g);
        add_edge(1, 2, g);
        add_edge(2, 3, g);
        add_edge(2, 4, g);

        array_1d<int> edge_weights{1, 2, 3, 4, 5, 6, 7};

        auto adj_mat = undirected_graph_2_adjacency_matrix(g, edge_weights, -1);

        array_2d<int> ref_adj_mat = {{-1, 1,  2,  3,  4},
                                     {1,  -1, 5,  -1, -1},
                                     {2,  5,  -1, 6,  7},
                                     {3,  -1, 6,  -1, -1},
                                     {4,  -1, 7,  -1, -1}};

        BOOST_CHECK(ref_adj_mat == adj_mat);
        auto res = adjacency_matrix_2_undirected_graph(ref_adj_mat, -1);

        auto &g2 = res.first;
        auto &ew2 = res.second;

        BOOST_CHECK(ew2 == edge_weights);
        BOOST_CHECK(num_vertices(g) == num_vertices(g2));
        BOOST_CHECK(num_edges(g) == num_edges(g2));
        auto it1 = edges(g);
        auto it2 = edges(g);
        for (auto i1 = it1.first, i2 = it2.first; i1 != it1.second; i1++, i2++) {
            BOOST_CHECK(*i1 == *i2);
        }
    }

BOOST_AUTO_TEST_SUITE_END();