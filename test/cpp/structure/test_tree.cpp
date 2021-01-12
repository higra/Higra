/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xio.hpp"
#include "higra/graph.hpp"
#include "../test_utils.hpp"
#include <functional>

namespace tree {

    using namespace hg;
    using namespace std;

    struct _data {

        hg::tree t;

        _data() : t(array_1d<index_t>{5, 5, 6, 6, 6, 7, 7, 7}) {
            t.compute_children();
        }

    } data;

    TEST_CASE("tree ctr", "[tree]") {
        hg::tree t1;
        REQUIRE(hg::num_vertices(t1) == 0);

        array_1d<index_t> p2{5, 5, 6, 6, 6, 7, 7, 7};
        hg::tree t2(p2);
        REQUIRE(hg::num_vertices(t2) == 8);
        REQUIRE(p2.size() == 8);

        array_1d<index_t> p3{5, 5, 6, 6, 6, 7, 7, 7};
        hg::tree t3(std::move(p3));
        REQUIRE(hg::num_vertices(t2) == 8);
        REQUIRE(p3.size() == 0);
    }

    TEST_CASE("tree children", "[tree]") {
        hg::tree t(array_1d<index_t>{5, 5, 6, 6, 6, 7, 7, 7});
        REQUIRE(t.children_computed() == false);
        t.compute_children();
        REQUIRE(t.children_computed() == true);
        t.clear_children();
        REQUIRE(t.children_computed() == false);
    }

    TEST_CASE("tree sizes", "[tree]") {
        auto t = data.t;
        REQUIRE(hg::category(t) == tree_category::partition_tree);
        REQUIRE(hg::root(t) == 7);
        REQUIRE(hg::num_vertices(t) == 8);
        REQUIRE(hg::num_edges(t) == 7);
        REQUIRE(hg::num_leaves(t) == 5);

        REQUIRE(hg::num_children(6, t) == 3);
        array_1d<index_t> vertices{5, 7, 6};
        array_1d<size_t> ref_numchildren{2, 2, 3};
        REQUIRE((hg::num_children(vertices, t) == ref_numchildren));

        REQUIRE(hg::is_leaf(4, t));
        REQUIRE(!hg::is_leaf(5, t));
        array_1d<index_t> vertices2{0, 5, 2, 3, 7};
        array_1d<bool> ref_isleaf{true, false, true, true, false};
        REQUIRE((hg::is_leaf(vertices2, t) == ref_isleaf));

        REQUIRE(hg::parent(4, t) == 6);
        array_1d<index_t> vertices3{0, 5, 2, 3, 7};
        array_1d<index_t> ref_parent{5, 7, 6, 6, 7};
        REQUIRE((hg::parent(vertices3, t) == ref_parent));
    }

    TEST_CASE("tree vertex iterator", "[tree]") {
        auto t = data.t;
        vector<hg::index_t> ref = {0, 1, 2, 3, 4, 5, 6, 7};
        for (auto v: vertex_iterator(t)) {
            REQUIRE(v == ref[v]);
        }
    }

    TEST_CASE("tree ancestors iterator", "[tree]") {
        auto t = data.t;
        vector<hg::index_t> ref = {1, 5, 7};
        REQUIRE(rangeEqual(ancestors_iterator(1, t), ref));

        vector<hg::index_t> ref2 = {6, 7};
        REQUIRE(rangeEqual(ancestors_iterator(6, t), ref2));

        vector<hg::index_t> ref3 = {7};
        REQUIRE(rangeEqual(ancestors_iterator(7, t), ref3));
    }

    TEST_CASE("tree degree", "[tree]") {
        auto t = data.t;
        vector<size_t> ref = {1, 1, 1, 1, 1, 3, 4, 2};
        for (auto v: vertex_iterator(t)) {
            REQUIRE(degree(v, t) == ref[v]);
            REQUIRE(in_degree(v, t) == ref[v]);
            REQUIRE(out_degree(v, t) == ref[v]);
        }
    }

    TEST_CASE("tree constructor asserts", "[tree]") {
        REQUIRE_THROWS_AS(hg::tree(xt::xarray<index_t>{5, 0, 6, 6, 6, 7, 7, 7}), std::runtime_error);
        REQUIRE_THROWS_AS(hg::tree(xt::xarray<index_t>{5, 1, 6, 6, 6, 7, 7, 7}), std::runtime_error);
        REQUIRE_THROWS_AS(hg::tree(xt::xarray<index_t>{5, 1, 6, 6, 6, 7, 7, 2}), std::runtime_error);
        REQUIRE_THROWS_AS(hg::tree(xt::xarray<index_t>{2, 2, 4, 4, 4}), std::runtime_error);
    }

    TEST_CASE("tree edge iterator", "[tree]") {
        auto g = data.t;

        vector<pair<index_t, index_t>> eref{
                {0, 5},
                {1, 5},
                {2, 6},
                {3, 6},
                {4, 6},
                {5, 7},
                {6, 7}
        };
        vector<pair<index_t, index_t>> etest;
        for (auto e: hg::edge_iterator(g)) {
            etest.push_back({source(e, g), target(e, g)});
        }

        REQUIRE(vectorEqual(eref, etest));
    }

    TEST_CASE("tree adjacent vertex iterator", "[tree]") {
        auto g = data.t;

        vector<vector<index_t>> adjListsRef{
                {5},
                {5},
                {6},
                {6},
                {6},
                {7, 0, 1},
                {7, 2, 3, 4},
                {5, 6}
        };
        vector<vector<index_t>> adjListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            adjListsTest.push_back(vector<index_t>());
            for (auto av: hg::adjacent_vertex_iterator(v, g)) {
                adjListsTest[v].push_back(av);
            }
            REQUIRE(vectorEqual(adjListsRef[v], adjListsTest[v]));
        }
    }

    TEST_CASE("tree out edge iterator", "[tree]") {
        auto g = data.t;

        vector<vector<pair<index_t, index_t>>> outListsRef{
                {{0, 5}},
                {{1, 5}},
                {{2, 6}},
                {{3, 6}},
                {{4, 6}},
                {{5, 7}, {5, 0}, {5, 1}},
                {{6, 7}, {6, 2}, {6, 3}, {6, 4}},
                {{7, 5}, {7, 6}}
        };
        vector<vector<pair<index_t, index_t>>> outListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            outListsTest.push_back(vector<pair<index_t, index_t>>());
            for (auto e: hg::out_edge_iterator(v, g)) {
                //showTypeName<decltype(e)>("edge type : ");
                outListsTest[v].push_back({source(e, g), target(e, g)});
            }
            REQUIRE(vectorEqual(outListsRef[v], outListsTest[v]));
        }

    }

    TEST_CASE("tree in edge iterator", "[tree]") {
        auto g = data.t;

        vector<vector<pair<index_t, index_t>>> outListsRef{
                {{5, 0}},
                {{5, 1}},
                {{6, 2}},
                {{6, 3}},
                {{6, 4}},
                {{7, 5}, {0, 5}, {1, 5}},
                {{7, 6}, {2, 6}, {3, 6}, {4, 6}},
                {{5, 7}, {6, 7}}
        };
        vector<vector<pair<index_t, index_t>>> outListsTest;

        for (auto v: hg::vertex_iterator(g)) {
            outListsTest.push_back(vector<pair<index_t, index_t>>());
            for (auto e: hg::in_edge_iterator(v, g)) {
                //showTypeName<decltype(e)>("edge type : ");
                outListsTest[v].push_back({source(e, g), target(e, g)});
            }
            REQUIRE(vectorEqual(outListsRef[v], outListsTest[v]));
        }

    }

    TEST_CASE("tree edge index iterator", "[tree]") {
        auto g = data.t;

        vector<index_t> ref{0, 1, 2, 3, 4, 5, 6};
        vector<index_t> test;

        for (auto v: hg::edge_iterator(g)) {
            test.push_back(v);
        }
        REQUIRE(vectorEqual(ref, test));
    }

    TEST_CASE("tree out edge index iterator", "[tree]") {
        auto g = data.t;

        vector<vector<index_t>> ref{
                {0},
                {1},
                {2},
                {3},
                {4},
                {5, 0, 1},
                {6, 2, 3, 4},
                {5, 6}
        };
        vector<vector<index_t>> test;

        for (auto v: hg::vertex_iterator(g)) {
            test.push_back(vector<index_t>());
            for (auto av: hg::out_edge_iterator(v, g)) {
                test[v].push_back(av);
            }
            REQUIRE(vectorEqual(ref[v], test[v]));
        }
    }

    TEST_CASE("tree in edge index iterator", "[tree]") {
        auto g = data.t;

        vector<vector<index_t>> ref{
                {0},
                {1},
                {2},
                {3},
                {4},
                {5, 0, 1},
                {6, 2, 3, 4},
                {5, 6}
        };
        vector<vector<index_t>> test;

        for (auto v: hg::vertex_iterator(g)) {
            test.push_back(vector<index_t>());
            for (auto av: hg::in_edge_iterator(v, g)) {
                test[v].push_back(av);
            }
            REQUIRE(vectorEqual(ref[v], test[v]));
        }
    }

    TEST_CASE("tree edge index", "[tree]") {
        auto g = data.t;

        vector<pair<index_t, index_t>> eref{
                {0, 5},
                {1, 5},
                {2, 6},
                {3, 6},
                {4, 6},
                {5, 7},
                {6, 7}
        };
        vector<pair<index_t, index_t>> etest;
        for (auto e: hg::edge_iterator(g)) {
            etest.push_back(edge_from_index(index(e, g), g));
        }

        REQUIRE(vectorEqual(eref, etest));
    }

    TEST_CASE("tree children iterator", "[tree]") {
        auto g = data.t;

        vector<vector<index_t>> ref{
                {},
                {},
                {},
                {},
                {},
                {0, 1},
                {2, 3, 4},
                {5, 6}
        };
        vector<vector<index_t>> test;

        for (auto v: hg::vertex_iterator(g)) {
            test.push_back(vector<index_t>());
            for (auto av: hg::children_iterator(v, g)) {
                test[v].push_back(av);
            }
            REQUIRE(vectorEqual(ref[v], test[v]));
        }

        REQUIRE(child(1, 5, g) == 1);

        array_1d<index_t> vertices{5, 7, 6};

        array_1d<index_t> ref_child0{0, 5, 2};
        REQUIRE((child(0, vertices, g) == ref_child0));

        array_1d<index_t> ref_child1{1, 6, 3};
        REQUIRE((child(1, vertices, g) == ref_child1));
    }

    TEST_CASE("tree tree topological order iterator", "[tree]") {
        auto tree = data.t;

        vector<index_t> ref1{0, 1, 2, 3, 4, 5, 6, 7};
        vector<index_t> t;
        for (auto v: hg::leaves_to_root_iterator(tree)) {
            t.push_back(v);
        }
        REQUIRE(vectorEqual(ref1, t));
        t.clear();

        vector<index_t> ref2{0, 1, 2, 3, 4, 5, 6};
        for (auto v: hg::leaves_to_root_iterator(tree, hg::leaves_it::include, hg::root_it::exclude)) {
            t.push_back(v);
        }
        REQUIRE(vectorEqual(ref2, t));
        t.clear();

        vector<index_t> ref3{5, 6, 7};
        for (auto v: hg::leaves_to_root_iterator(tree, hg::leaves_it::exclude, hg::root_it::include)) {
            t.push_back(v);
        }
        REQUIRE(vectorEqual(ref3, t));
        t.clear();

        vector<index_t> ref4{5, 6};
        for (auto v: hg::leaves_to_root_iterator(tree, hg::leaves_it::exclude, hg::root_it::exclude)) {
            t.push_back(v);
        }
        REQUIRE(vectorEqual(ref4, t));
        t.clear();
    }

    TEST_CASE("tree reverse topological order iterator", "[tree]") {
        auto tree = data.t;

        vector<index_t> ref1{7, 6, 5, 4, 3, 2, 1, 0};
        vector<index_t> t;
        for (auto v: hg::root_to_leaves_iterator(tree)) {
            t.push_back(v);
        }
        REQUIRE(vectorEqual(ref1, t));
        t.clear();

        vector<index_t> ref2{6, 5, 4, 3, 2, 1, 0};
        for (auto v: hg::root_to_leaves_iterator(tree, hg::leaves_it::include, hg::root_it::exclude)) {
            t.push_back(v);
        }
        REQUIRE(vectorEqual(ref2, t));
        t.clear();

        vector<index_t> ref3{7, 6, 5};
        for (auto v: hg::root_to_leaves_iterator(tree, hg::leaves_it::exclude, hg::root_it::include)) {
            t.push_back(v);
        }
        REQUIRE(vectorEqual(ref3, t));
        t.clear();

        vector<index_t> ref4{6, 5};
        for (auto v: hg::root_to_leaves_iterator(tree, hg::leaves_it::exclude, hg::root_it::exclude)) {
            t.push_back(v);
        }
        REQUIRE(vectorEqual(ref4, t));
        t.clear();
    }

    TEST_CASE("tree adjacency matrix", "[tree]") {
        auto t = data.t;

        array_1d<int> edge_weights{1, 2, 3, 4, 5, 6, 7};

        auto adj_mat = undirected_graph_2_adjacency_matrix(t, edge_weights);

        array_2d<int> ref_adj_mat = {
                {0, 0, 0, 0, 0, 1, 0, 0},
                {0, 0, 0, 0, 0, 2, 0, 0},
                {0, 0, 0, 0, 0, 0, 3, 0},
                {0, 0, 0, 0, 0, 0, 4, 0},
                {0, 0, 0, 0, 0, 0, 5, 0},
                {1, 2, 0, 0, 0, 0, 0, 6},
                {0, 0, 3, 4, 5, 0, 0, 7},
                {0, 0, 0, 0, 0, 6, 7, 0}
        };

        REQUIRE((ref_adj_mat == adj_mat));
    }

    TEST_CASE("tree find_region", "[tree]") {
        hg::tree t(array_1d<index_t>{8, 8, 9, 7, 7, 11, 11, 9, 10, 10, 12, 12, 12});

        array_1d<double> altitudes{0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 2, 3};

        array_1d<index_t> vertices{0, 0, 0, 2, 2, 9, 9, 12};
        array_1d<double> lambdas{2, 3, 4, 1, 2, 2, 3, 3};

        array_1d<index_t> expected_results{0, 10, 12, 2, 9, 9, 10, 12};

        for (index_t i = 0; i < (index_t) vertices.size(); i++) {
            REQUIRE((find_region(vertices(i), lambdas(i), altitudes, t) == expected_results(i)));
        }

        REQUIRE((find_region(vertices, lambdas, altitudes, t) == expected_results));
    }

    TEST_CASE("test lca with altitudes pairs of vertices", "[tree]") {
        hg::tree t(xt::xarray<index_t>{5, 5, 6, 6, 6, 7, 7, 7});
        REQUIRE(hg::lowest_common_ancestor(0, 0, t) == 0);
        REQUIRE(hg::lowest_common_ancestor(3, 3, t) == 3);
        REQUIRE(hg::lowest_common_ancestor(5, 5, t) == 5);
        REQUIRE(hg::lowest_common_ancestor(7, 7, t) == 7);
        REQUIRE(hg::lowest_common_ancestor(0, 1, t) == 5);
        REQUIRE(hg::lowest_common_ancestor(1, 0, t) == 5);
        REQUIRE(hg::lowest_common_ancestor(2, 3, t) == 6);
        REQUIRE(hg::lowest_common_ancestor(2, 4, t) == 6);
        REQUIRE(hg::lowest_common_ancestor(3, 4, t) == 6);
        REQUIRE(hg::lowest_common_ancestor(5, 6, t) == 7);
        REQUIRE(hg::lowest_common_ancestor(0, 2, t) == 7);
        REQUIRE(hg::lowest_common_ancestor(1, 4, t) == 7);
        REQUIRE(hg::lowest_common_ancestor(2, 6, t) == 6);
    }

    TEST_CASE("test lca with altitudes vectorial", "[tree]") {
        hg::tree t(xt::xarray<index_t>{5, 5, 6, 6, 6, 7, 7, 7});
        array_1d<index_t> v1{0, 0, 1, 3};
        array_1d<index_t> v2{0, 3, 0, 0};

        auto l = hg::lowest_common_ancestor(v1, v2, t);

        array_1d<index_t> ref{0, 7, 5, 7};
        REQUIRE((l == ref));
    }

    TEST_CASE("edge lists") {
        auto g = data.t;

        array_1d<index_t> sources_ref{0, 1, 2, 3, 4, 5, 6};
        array_1d<index_t> targets_ref{5, 5, 6, 6, 6, 7, 7};

        auto src = sources(g);
        REQUIRE((sources_ref == src));

        auto tgt = targets(g);
        REQUIRE((targets_ref == tgt));
    }
}