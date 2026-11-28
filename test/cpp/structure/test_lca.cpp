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
#include "higra/image/graph_image.hpp"
#include "higra/structure/lca_fast.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "../test_utils.hpp"

namespace lca {

    using namespace hg;
    using namespace std;

    struct _data {

        hg::tree t;

        _data() : t(hg::array<index_t>{5, 5, 6, 6, 6, 7, 7, 7}) {
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
        hg_test::random_seed(42);
        auto g = hg::get_4_adjacency_graph({20, 20});
        auto w = hg_test::random_uniform<double>({num_edges(g)});
        auto h = hg::bpt_canonical(g, w);
        auto &tree = h.tree;

        TestType lca(tree);
        for (index_t i = 0; i < num_vertices(g); ++i) {
            for (index_t j = i; j < num_vertices(g); j++) {
                auto ref = lowest_common_ancestor(i, j, tree);
                auto r1 = lca.lca(i, j);
                auto r2 = lca.lca(j, i);
                REQUIRE(ref == r1);
                REQUIRE(ref == r2);
            }
        }
    }

    TEST_CASE("lca sparse table block", "[lca]") {
        hg_test::random_seed(42);
        auto g = hg::get_4_adjacency_graph({10, 10});
        auto w = hg_test::random_uniform<double>({num_edges(g)});
        auto h = hg::bpt_canonical(g, w);
        auto &tree = h.tree;

        hg::lca_sparse_table_block lca(tree, 4);
        for (index_t i = 0; i < num_vertices(g); ++i) {
            for (index_t j = i; j < num_vertices(g); j++) {
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

    TEST_CASE("lca state round trip", "[lca]") {
        // same tree and expected state as the pickling fixture of the Python tests (baseline build)
        tree t(array_1d<index_t>{3, 3, 4, 4, 4});
        array_1d<index_t> map_ref{4, 3, 1, 3, 0, 3, 4, 2, 4};
        array_1d<index_t> depth_ref{0, 1, 2, 1, 2, 1, 0, 1, 0};
        array_1d<index_t> first_ref{4, 2, 7, 1, 0};

        auto check_all_pairs = [&t](const auto &l) {
            for (index_t i = 0; i < 5; ++i) {
                for (index_t j = 0; j < 5; ++j) {
                    REQUIRE(l.lca(i, j) == lowest_common_ancestor(i, j, t));
                }
            }
        };

        SECTION("sparse table") {
            lca_sparse_table lca(t);
            auto state = lca.get_state();
            static_assert(std::is_same<decltype(state), lca_sparse_table::internal_state>::value,
                          "the internal state is a plain structure");
            static_assert(std::is_same<decltype(state.tree_Euler_tour_map), array_1d<index_t>>::value, "");
            REQUIRE((state.tree_Euler_tour_map == map_ref));
            REQUIRE((state.tree_Euler_tour_depth == depth_ref));
            REQUIRE((state.first_visit_in_Euler_tour == first_ref));
            auto &sp = state.rmq_state.sparse_table;
            REQUIRE(sp.size() == 4);
            REQUIRE((sp[0] == array_1d<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8}));
            REQUIRE((sp[1] == array_1d<std::size_t>{0, 1, 3, 3, 5, 6, 6, 8}));
            REQUIRE((sp[2] == array_1d<std::size_t>{0, 3, 5, 6, 6, 8}));
            REQUIRE((sp[3] == array_1d<std::size_t>{6, 8}));

            auto lca2 = lca_sparse_table::make_from_state(state);
            // the rebuilt object owns copies: changing the state afterwards has no effect
            state.tree_Euler_tour_map.fill(0);
            state.rmq_state.sparse_table[1].fill(0);
            check_all_pairs(lca2);
            auto state2 = lca2.get_state();
            REQUIRE((state2.tree_Euler_tour_map == map_ref));
            REQUIRE((state2.rmq_state.sparse_table[1] == array_1d<std::size_t>{0, 1, 3, 3, 5, 6, 6, 8}));

            auto lca3 = lca_sparse_table::make_from_state(std::move(state2));
            check_all_pairs(lca3);
            REQUIRE(lca3.num_elements() == 5);
        }

        SECTION("sparse table block") {
            lca_sparse_table_block lca(t, 2);
            auto state = lca.get_state();
            static_assert(std::is_same<decltype(state), lca_sparse_table_block::internal_state>::value, "");
            REQUIRE((state.tree_Euler_tour_map == map_ref));
            REQUIRE(state.rmq_state.data_size == 9);
            REQUIRE(state.rmq_state.block_size == 2);
            REQUIRE(state.rmq_state.num_blocks == 5);
            REQUIRE((state.rmq_state.block_minimum_prefix == array_1d<index_t>{0, 0, 2, 3, 4, 5, 6, 6, 8, 8}));
            REQUIRE((state.rmq_state.block_minimum_suffix == array_1d<index_t>{0, 1, 3, 3, 5, 5, 6, 7, 9, 9}));
            auto &sp = state.rmq_state.sparse_table.sparse_table;
            REQUIRE(sp.size() == 3);
            REQUIRE((sp[0] == array_1d<std::size_t>{0, 3, 5, 6, 8}));
            REQUIRE((sp[2] == array_1d<std::size_t>{6, 8}));

            auto lca2 = lca_sparse_table_block::make_from_state(state);
            state.rmq_state.block_minimum_prefix.fill(0);
            check_all_pairs(lca2);
            auto lca3 = lca_sparse_table_block::make_from_state(lca2.get_state());
            check_all_pairs(lca3);
            REQUIRE((lca3.get_state().rmq_state.block_minimum_prefix ==
                     array_1d<index_t>{0, 0, 2, 3, 4, 5, 6, 6, 8, 8}));
        }
    }

    template<typename lca_t>
    void check_lca_of_tree(const lca_t &lca, const tree &t) {
        for (index_t i = 0; i < num_vertices(t); ++i) {
            for (index_t j = i; j < num_vertices(t); j++) {
                REQUIRE(lca.lca(i, j) == lowest_common_ancestor(i, j, t));
            }
        }
    }

    TEMPLATE_TEST_CASE("lca copy and move", "[lca]", hg::lca_sparse_table, hg::lca_sparse_table_block) {
        static_assert(std::is_nothrow_move_constructible<TestType>::value, "");
        static_assert(std::is_nothrow_move_assignable<TestType>::value, "");
        hg_test::random_seed(42);
        auto g = hg::get_4_adjacency_graph({10, 10});
        auto t1 = hg::bpt_canonical(g, hg_test::random_uniform<double>({num_edges(g)})).tree;
        auto t2 = hg::bpt_canonical(g, hg_test::random_uniform<double>({num_edges(g)})).tree;

        SECTION("copy constructor") {
            TestType source(t1);
            TestType copy(source);
            // replacing the source frees its Euler tour, which the copy must not read
            source = TestType(t2);
            check_lca_of_tree(copy, t1);
            check_lca_of_tree(source, t2);
        }

        SECTION("copy assignment") {
            TestType source(t1);
            TestType copy(t2);
            copy = source;
            source = TestType(t2);
            check_lca_of_tree(copy, t1);
            check_lca_of_tree(source, t2);
            const TestType &same = copy;
            copy = same;
            check_lca_of_tree(copy, t1);
        }

        SECTION("move") {
            TestType source(t1);
            TestType moved(std::move(source));
            check_lca_of_tree(moved, t1);
            TestType assigned(t2);
            assigned = std::move(moved);
            check_lca_of_tree(assigned, t1);
        }
    }

    TEST_CASE("range minimum query on empty input", "[lca]") {
        using namespace hg::range_minimum_query_internal;
        auto empty = array_1d<index_t>::from_shape({0});
        REQUIRE_NOTHROW(rmq_sparse_table<index_t>(empty));
        REQUIRE_NOTHROW(rmq_sparse_table_block<index_t>(empty, 4));

        array_1d<index_t> one{3};
        REQUIRE_NOTHROW(rmq_sparse_table<index_t>(one));
        REQUIRE_NOTHROW(rmq_sparse_table_block<index_t>(one, 4));
    }
}
