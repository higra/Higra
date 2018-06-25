//
// Created by user on 3/22/18.
//

#include <boost/test/unit_test.hpp>
#include "xtensor/xio.hpp"
#include "higra/graph.hpp"
#include "test_utils.hpp"
#include <functional>

BOOST_AUTO_TEST_SUITE(treeGraph);

    using namespace hg;
    using namespace std;


    struct _data {

        hg::tree t;

        _data() : t(xt::xarray<long>{5, 5, 6, 6, 6, 7, 7, 7}) {
        }

    } data;


    BOOST_AUTO_TEST_CASE(sizeTree) {
        auto t = data.t;
        BOOST_CHECK(t.root() == 7);
        BOOST_CHECK(t.num_vertices() == 8);
        BOOST_CHECK(t.num_edges() == 7);
        BOOST_CHECK(t.num_leaves() == 5);
    }

    BOOST_AUTO_TEST_CASE(vertexIteratorTree) {
        auto t = data.t;
        vector<ulong> ref = {0, 1, 2, 3, 4, 5, 6, 7};
        for (auto v: vertex_iterator(t)) {
            BOOST_CHECK(v == ref[v]);
        }
    }


    BOOST_AUTO_TEST_CASE(degreeTree) {
        auto t = data.t;
        vector<ulong> ref = {1, 1, 1, 1, 1, 3, 4, 2};
        for (auto v: vertex_iterator(t)) {
            BOOST_CHECK(degree(v, t) == ref[v]);
            BOOST_CHECK(in_degree(v, t) == ref[v]);
            BOOST_CHECK(out_degree(v, t) == ref[v]);
        }

    }

    BOOST_AUTO_TEST_CASE(treeFail) {
        BOOST_REQUIRE_THROW(hg::tree(xt::xarray<ulong>{5, 0, 6, 6, 6, 7, 7, 7}), std::runtime_error);
        BOOST_REQUIRE_THROW(hg::tree(xt::xarray<ulong>{5, 1, 6, 6, 6, 7, 7, 7}), std::runtime_error);
        BOOST_REQUIRE_THROW(hg::tree(xt::xarray<ulong>{5, 1, 6, 6, 6, 7, 7, 2}), std::runtime_error);
        BOOST_REQUIRE_THROW(hg::tree(xt::xarray<ulong>{2, 2, 4, 4, 4}), std::runtime_error);
    }

    BOOST_AUTO_TEST_CASE(edgeIteratorTree) {

        auto g = data.t;

        vector<pair<ulong, ulong>> eref{{0, 5},
                                        {1, 5},
                                        {2, 6},
                                        {3, 6},
                                        {4, 6},
                                        {5, 7},
                                        {6, 7}};
        vector<pair<ulong, ulong>> etest;
        for (auto e: hg::edge_iterator(g)) {
            etest.push_back({source(e, g), target(e, g)});
        }

        BOOST_CHECK(vectorEqual(eref, etest));
    }

    BOOST_AUTO_TEST_CASE(adjacentVertexIteratorTreeGraph) {

        auto g = data.t;

        vector<vector<ulong>> adjListsRef{{5},
                                          {5},
                                          {6},
                                          {6},
                                          {6},
                                          {7, 0, 1},
                                          {7, 2, 3, 4},
                                          {5, 6}};
        vector<vector<ulong>> adjListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            adjListsTest.push_back(vector<ulong>());
            for (auto av: hg::adjacent_vertex_iterator(v, g)) {
                adjListsTest[v].push_back(av);
            }
            BOOST_CHECK(vectorEqual(adjListsRef[v], adjListsTest[v]));
        }
    }

    BOOST_AUTO_TEST_CASE(outEdgeIteratorTreeGraph) {

        auto g = data.t;

        vector<vector<pair<ulong, ulong>>> outListsRef{{{0, 5}},
                                                       {{1, 5}},
                                                       {{2, 6}},
                                                       {{3, 6}},
                                                       {{4, 6}},
                                                       {{5, 7}, {5, 0}, {5, 1}},
                                                       {{6, 7}, {6, 2}, {6, 3}, {6, 4}},
                                                       {{7, 5}, {7, 6}}};
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

    BOOST_AUTO_TEST_CASE(inEdgeIteratorTreeGraph) {

        auto g = data.t;

        vector<vector<pair<ulong, ulong>>> outListsRef{{{5, 0}},
                                                       {{5, 1}},
                                                       {{6, 2}},
                                                       {{6, 3}},
                                                       {{6, 4}},
                                                       {{7, 5}, {0, 5}, {1, 5}},
                                                       {{7, 6}, {2, 6}, {3, 6}, {4, 6}},
                                                       {{5, 7}, {6, 7}}};
        vector<vector<pair<ulong, ulong>>> outListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            outListsTest.push_back(vector<pair<ulong, ulong>>());
            for (auto e: hg::in_edge_iterator(v, g)) {
                //showTypeName<decltype(e)>("edge type : ");
                outListsTest[v].push_back({source(e, g), target(e, g)});
            }
            BOOST_CHECK(vectorEqual(outListsRef[v], outListsTest[v]));
        }

    }

    BOOST_AUTO_TEST_CASE(edgeIndexIteratorTreeGraph) {

        auto g = data.t;

        vector<ulong> ref{0, 1, 2, 3, 4, 5, 6};
        vector<ulong> test;

        for (auto v: hg::edge_index_iterator(g)) {
            test.push_back(v);
        }
        BOOST_CHECK(vectorEqual(ref, test));
    }

    BOOST_AUTO_TEST_CASE(outEdgeIndexIteratorTreeGraph) {

        auto g = data.t;

        vector<vector<ulong>> ref{{0},
                                  {1},
                                  {2},
                                  {3},
                                  {4},
                                  {5, 0, 1},
                                  {6, 2, 3, 4},
                                  {5, 6}};
        vector<vector<ulong>> test;

        for (auto v: hg::vertex_iterator(g)) {
            test.push_back(vector<ulong>());
            for (auto av: hg::out_edge_index_iterator(v, g)) {
                test[v].push_back(av);
            }
            BOOST_CHECK(vectorEqual(ref[v], test[v]));
        }
    }

    BOOST_AUTO_TEST_CASE(inEdgeIndexIteratorTreeGraph) {

        auto g = data.t;

        vector<vector<ulong>> ref{{0},
                                  {1},
                                  {2},
                                  {3},
                                  {4},
                                  {5, 0, 1},
                                  {6, 2, 3, 4},
                                  {5, 6}};
        vector<vector<ulong>> test;

        for (auto v: hg::vertex_iterator(g)) {
            test.push_back(vector<ulong>());
            for (auto av: hg::in_edge_index_iterator(v, g)) {
                test[v].push_back(av);
            }
            BOOST_CHECK(vectorEqual(ref[v], test[v]));
        }
    }

    BOOST_AUTO_TEST_CASE(edgeIndex) {

        auto g = data.t;

        vector<pair<ulong, ulong>> eref{{0, 5},
                                        {1, 5},
                                        {2, 6},
                                        {3, 6},
                                        {4, 6},
                                        {5, 7},
                                        {6, 7}};
        vector<pair<ulong, ulong>> etest;
        for (auto ei: hg::edge_index_iterator(g)) {
            etest.push_back(edge(ei, g));
        }

        BOOST_CHECK(vectorEqual(eref, etest));
    }

    BOOST_AUTO_TEST_CASE(childrenIteratorTreeGraph) {

        auto g = data.t;

        vector<vector<ulong>> ref{{},
                                  {},
                                  {},
                                  {},
                                  {},
                                  {0, 1},
                                  {2, 3, 4},
                                  {5, 6}};
        vector<vector<ulong>> test;

        for (auto v: hg::vertex_iterator(g)) {
            test.push_back(vector<ulong>());
            for (auto av: hg::children_iterator(v, g)) {
                test[v].push_back(av);
            }
            BOOST_CHECK(vectorEqual(ref[v], test[v]));
        }
    }

    BOOST_AUTO_TEST_CASE(treeTopologicalIterator) {

        auto tree = data.t;

        vector<ulong> ref1{0, 1, 2, 3, 4, 5, 6, 7};
        vector<ulong> t;
        for (auto v: tree.iterate_from_leaves_to_root()) {
            t.push_back(v);
        }
        BOOST_CHECK(vectorEqual(ref1, t));
        t.clear();

        vector<ulong> ref2{0, 1, 2, 3, 4, 5, 6};
        for (auto v: tree.iterate_from_leaves_to_root(hg::leaves_it::include, hg::root_it::exclude)) {
            t.push_back(v);
        }
        BOOST_CHECK(vectorEqual(ref2, t));
        t.clear();

        vector<ulong> ref3{5, 6, 7};
        for (auto v: tree.iterate_from_leaves_to_root(hg::leaves_it::exclude, hg::root_it::include)) {
            t.push_back(v);
        }
        BOOST_CHECK(vectorEqual(ref3, t));
        t.clear();

        vector<ulong> ref4{5, 6};
        for (auto v: tree.iterate_from_leaves_to_root(hg::leaves_it::exclude, hg::root_it::exclude)) {
            t.push_back(v);
        }
        BOOST_CHECK(vectorEqual(ref4, t));
        t.clear();
    }

    BOOST_AUTO_TEST_CASE(treeRevTopologicalIterator) {

        auto tree = data.t;

        vector<ulong> ref1{7, 6, 5, 4, 3, 2, 1, 0};
        vector<ulong> t;
        for (auto v: tree.iterate_from_root_to_leaves()) {
            t.push_back(v);
        }
        BOOST_CHECK(vectorEqual(ref1, t));
        t.clear();

        vector<ulong> ref2{6, 5, 4, 3, 2, 1, 0};
        for (auto v: tree.iterate_from_root_to_leaves(hg::leaves_it::include, hg::root_it::exclude)) {
            t.push_back(v);
        }
        BOOST_CHECK(vectorEqual(ref2, t));
        t.clear();

        vector<ulong> ref3{7, 6, 5};
        for (auto v: tree.iterate_from_root_to_leaves(hg::leaves_it::exclude, hg::root_it::include)) {
            t.push_back(v);
        }
        BOOST_CHECK(vectorEqual(ref3, t));
        t.clear();

        vector<ulong> ref4{6, 5};
        for (auto v: tree.iterate_from_root_to_leaves(hg::leaves_it::exclude, hg::root_it::exclude)) {
            t.push_back(v);
        }
        BOOST_CHECK(vectorEqual(ref4, t));
        t.clear();
    }


BOOST_AUTO_TEST_SUITE_END();