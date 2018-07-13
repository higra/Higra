//
// Created by user on 3/9/18.
//

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include "higra/graph.hpp"
#include "test_utils.hpp"


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
        array_2d<index_t> ref{{2, 0},
                              {2, 2}};

        BOOST_CHECK(xt::allclose(degree(indices, g), ref));
        BOOST_CHECK(xt::allclose(in_degree(indices, g), ref));
        BOOST_CHECK(xt::allclose(out_degree(indices, g), ref));
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(copyCTR, T, test_types) {
        auto g0 = data<T>::g();
        auto g = hg::copy_graph<T>(g0);

        vector<pair<unsigned long, unsigned long>> eref{{0, 1},
                                                        {1, 2},
                                                        {0, 2}};
        vector<pair<unsigned long, unsigned long>> etest;
        for (auto e: hg::edge_iterator(g)) {
            etest.push_back({source(e, g), target(e, g)});
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
            BOOST_CHECK(vectorSame(outListsRef[v], outListsTest[v]));
            BOOST_CHECK(out_degree(v, g) == outListsRef[v].size());

        }


    }


    BOOST_AUTO_TEST_CASE_TEMPLATE(vertexIteratorSimpleGraph, T, test_types) {

        auto g = data<T>::g();

        vector<unsigned long> vref{0, 1, 2, 3};
        vector<unsigned long> vtest;

        for (auto v: hg::vertex_iterator(g))
            vtest.push_back(v);

        BOOST_CHECK(vectorEqual(vref, vtest));

    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(edgeIteratorSimpleGraph, T, test_types) {

        auto g = data<T>::g();

        vector<pair<unsigned long, unsigned long>> eref{{0, 1},
                                                        {1, 2},
                                                        {0, 2}};
        vector<pair<unsigned long, unsigned long>> etest;
        for (auto e: hg::edge_iterator(g)) {
            etest.push_back({source(e, g), target(e, g)});
        }

        BOOST_CHECK(vectorEqual(eref, etest));
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(outEdgeIteratorSimpleGraph, T, test_types) {

        auto g = data<T>::g();

        vector<vector<pair<unsigned long, unsigned long>>> outListsRef{{{0, 1}, {0, 2}},
                                                                       {{1, 0}, {1, 2}},
                                                                       {{2, 1}, {2, 0}},
                                                                       {}};
        vector<vector<pair<unsigned long, unsigned long>>> outListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            outListsTest.push_back(vector<pair<unsigned long, unsigned long>>());
            for (auto e: hg::out_edge_iterator(v, g)) {
                //showTypeName<decltype(e)>("edge type : ");
                outListsTest[v].push_back({source(e, g), target(e, g)});
            }
            BOOST_CHECK(vectorSame(outListsRef[v], outListsTest[v]));
        }

    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(inEdgeIteratorSimpleGraph, T, test_types) {

        auto g = data<T>::g();

        vector<vector<pair<unsigned long, unsigned long>>> inListsRef{{{1, 0}, {2, 0}},
                                                                      {{0, 1}, {2, 1}},
                                                                      {{1, 2}, {0, 2}},
                                                                      {}};
        vector<vector<pair<unsigned long, unsigned long>>> inListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            inListsTest.push_back(vector<pair<unsigned long, unsigned long>>());
            for (auto e: hg::in_edge_iterator(v, g))
                inListsTest[v].push_back({source(e, g), target(e, g)});
            BOOST_CHECK(vectorSame(inListsRef[v], inListsTest[v]));
        }
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(adjacentVertexIteratorSimpleGraph, T, test_types) {

        auto g = data<T>::g();

        vector<vector<unsigned long>> adjListsRef{{1, 2},
                                                  {0, 2},
                                                  {1, 0},
                                                  {}};
        vector<vector<unsigned long>> adjListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            adjListsTest.push_back(vector<unsigned long>());
            for (auto av: hg::adjacent_vertex_iterator(v, g)) {
                adjListsTest[v].push_back(av);
            }
            BOOST_CHECK(vectorSame(adjListsRef[v], adjListsTest[v]));
        }
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(edgeIndexIteratorSimpleGraph, T, test_types) {

        auto g = data<T>::g();

        vector<unsigned long> ref{0, 1, 2};
        vector<unsigned long> test;

        for (auto v: hg::edge_index_iterator(g)) {
            test.push_back(v);
        }
        BOOST_CHECK(vectorSame(ref, test));
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(outEdgeIndexIteratorSimpleGraph, T, test_types) {

        auto g = data<T>::g();

        vector<vector<unsigned long>> ref{{0, 2},
                                          {0, 1},
                                          {1, 2},
                                          {}};
        vector<vector<unsigned long>> test;

        for (auto v: hg::vertex_iterator(g)) {
            test.push_back(vector<unsigned long>());
            for (auto av: hg::out_edge_index_iterator(v, g)) {
                test[v].push_back(av);
            }
            BOOST_CHECK(vectorSame(ref[v], test[v]));
        }
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(inEdgeIndexIteratorSimpleGraph, T, test_types) {

        auto g = data<T>::g();

        vector<vector<unsigned long>> ref{{0, 2},
                                          {0, 1},
                                          {1, 2},
                                          {}};
        vector<vector<unsigned long>> test;

        for (auto v: hg::vertex_iterator(g)) {
            test.push_back(vector<unsigned long>());
            for (auto av: hg::in_edge_index_iterator(v, g)) {
                test[v].push_back(av);
            }
            BOOST_CHECK(vectorSame(ref[v], test[v]));
        }
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(edgeIndex, T, test_types) {

        auto g = data<T>::g();

        vector<pair<unsigned long, unsigned long>> eref{{0, 1},
                                                        {1, 2},
                                                        {0, 2}};
        vector<pair<unsigned long, unsigned long>> etest;
        for (auto ei: hg::edge_index_iterator(g)) {
            etest.push_back(edge(ei, g));
        }

        BOOST_CHECK(vectorSame(eref, etest));
    }

    BOOST_AUTO_TEST_CASE_TEMPLATE(removeEdge, T, test_types) {

        auto g = data<T>::g();

        g.remove_edge(1);

        vector<pair<unsigned long, unsigned long>> eref{{0,             1}, // deleted {1,2}
                                        {                invalid_index, invalid_index},
                                                        {0,             2}};
        vector<pair<unsigned long, unsigned long>> etest;
        for (auto ei: hg::edge_index_iterator(g)) {
            etest.push_back(edge(ei, g));
        }

        BOOST_CHECK(vectorSame(eref, etest));

        BOOST_CHECK(degree(0, g) == 2);
        BOOST_CHECK(degree(1, g) == 1);
        BOOST_CHECK(degree(2, g) == 1);

        vector<vector<unsigned long>> adjListsRef{{1, 2},
                                                  {0},
                                                  {0},
                                                  {}};
        vector<vector<unsigned long>> adjListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            adjListsTest.push_back(vector<unsigned long>());
            for (auto av: hg::adjacent_vertex_iterator(v, g)) {
                adjListsTest[v].push_back(av);
            }
            BOOST_CHECK(vectorSame(adjListsRef[v], adjListsTest[v]));
        }
    }

BOOST_AUTO_TEST_SUITE_END();