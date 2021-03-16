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
#include "xtensor/xrandom.hpp"
#include "higra/graph.hpp"
#include "higra/image/graph_image.hpp"
#include "higra/structure/lca_fast.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "../test_utils.hpp"

namespace lca {

    using namespace hg;
    using namespace std;

    struct _data {

        hg::tree t;

        _data() : t(xt::xarray<index_t>{5, 5, 6, 6, 6, 7, 7, 7}) {
        }

    } data;


    TEMPLATE_TEST_CASE("lca pairs of vertices", "[lca]", hg::lca_sparse_table, hg::lca_sparse_table_block) {
        auto t = data.t;
        TestType lca(t);
        REQUIRE(lca.lca(0, 0) == 0);
        REQUIRE(lca.lca(3, 3) == 3);
        REQUIRE(lca.lca(5, 5) == 5);
        REQUIRE(lca.lca(7, 7) == 7);
        REQUIRE(lca.lca(0, 1) == 5);
        REQUIRE(lca.lca(1, 0) == 5);
        REQUIRE(lca.lca(2, 3) == 6);
        REQUIRE(lca.lca(2, 4) == 6);
        REQUIRE(lca.lca(3, 4) == 6);
        REQUIRE(lca.lca(5, 6) == 7);
        REQUIRE(lca.lca(0, 2) == 7);
        REQUIRE(lca.lca(1, 4) == 7);
        REQUIRE(lca.lca(2, 6) == 6);
    }

    TEMPLATE_TEST_CASE("lca iterators", "[lca]", hg::lca_sparse_table, hg::lca_sparse_table_block) {
        auto g = get_4_adjacency_graph({2, 2});
        tree t(array_1d<index_t>{4, 4, 5, 5, 6, 6, 6});
        TestType lca(t);
        auto l = lca.lca(edge_iterator(g));
        array_1d<index_t> ref{4, 6, 6, 5};
        REQUIRE((l == ref));
    }

    TEMPLATE_TEST_CASE("lca tensors", "[lca]", hg::lca_sparse_table, hg::lca_sparse_table_block) {
        tree t(array_1d<index_t>{4, 4, 5, 5, 6, 6, 6});
        TestType lca(t);
        array_1d<index_t> v1{0, 0, 1, 3};
        array_1d<index_t> v2{0, 3, 0, 0};
        auto l = lca.lca(v1, v2);
        array_1d<index_t> ref{0, 6, 4, 6};
        REQUIRE((l == ref));
    }

    TEMPLATE_TEST_CASE("lca sanity", "[lca]", hg::lca_sparse_table, hg::lca_sparse_table_block) {
        xt::random::seed(42);
        auto g = hg::get_4_adjacency_graph({20, 20});
        auto w = xt::eval(xt::random::rand<double>({num_edges(g)}));
        auto h = hg::bpt_canonical(g, w);
        auto &tree = h.tree;

        TestType lca(tree);
        for (index_t i = 0; i < (index_t)num_vertices(g); ++i) {
            for (index_t j = i; j < (index_t)num_vertices(g); j++) {
                auto ref = lowest_common_ancestor(i, j, tree);
                auto r1 = lca.lca(i, j);
                auto r2 = lca.lca(j, i);
                REQUIRE(ref == r1);
                REQUIRE(ref == r2);
            }
        }
    }

    TEST_CASE("lca sparse table block", "[lca]") {
        xt::random::seed(42);
        auto g = hg::get_4_adjacency_graph({10, 10});
        auto w = xt::eval(xt::random::rand<double>({num_edges(g)}));
        auto h = hg::bpt_canonical(g, w);
        auto &tree = h.tree;

        hg::lca_sparse_table_block lca(tree, 4);
        for (index_t i = 0; i < (index_t)num_vertices(g); ++i) {
            for (index_t j = i; j < (index_t)num_vertices(g); j++) {
                auto ref = lowest_common_ancestor(i, j, tree);
                auto r1 = lca.lca(i, j);
                auto r2 = lca.lca(j, i);
                REQUIRE(ref == r1);
                REQUIRE(ref == r2);
            }
        }
    }

    TEMPLATE_TEST_CASE("lca serialization", "[lca]", hg::lca_sparse_table, hg::lca_sparse_table_block) {
        tree t(array_1d<index_t>{4, 4, 5, 5, 6, 6, 6});
        TestType lca(t);
        array_1d<index_t> v1{0, 0, 1, 3};
        array_1d<index_t> v2{0, 3, 0, 0};

        auto state = lca.get_state();

        TestType lca2 = TestType::make_from_state(state);
        array_1d<index_t> ref{0, 6, 4, 6};
        auto l = lca2.lca(v1, v2);
        REQUIRE((l == ref));

        TestType lca3 = TestType::make_from_state(std::move(state));
        auto l2 = lca3.lca(v1, v2);
        REQUIRE((l2 == ref));
    }
}