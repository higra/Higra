/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/algo/alignment.hpp"
#include "higra/image/graph_image.hpp"
#include "../test_utils.hpp"
#include "higra/algo/tree.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

using namespace hg;

namespace test_alignment {

    TEST_CASE("project fine to coarse labelisation", "[alignment]") {

        array_1d<index_t> fine_labels{0, 1, 2, 3, 4, 2, 3, 4, 2};
        array_1d<index_t> coarse_labels{0, 1, 1, 0, 2, 2, 0, 2, 2};

        auto map = project_fine_to_coarse_labelisation(fine_labels, coarse_labels);

        array_1d<index_t> ref_map{0, 1, 2, 0, 2};
        REQUIRE((ref_map == map));
    }

    TEST_CASE("hierarchy alignement", "[alignment]") {

        auto g = get_4_adjacency_graph({3, 3});
        array_1d<int> fine_labels{0, 1, 2, 3, 4, 2, 3, 4, 2};

        auto aligner = make_hierarchy_aligner_from_labelisation(g, fine_labels);

        hg::tree t(array_1d<index_t>{9, 10, 10, 9, 11, 11, 9, 11, 11, 13, 12, 12, 13, 13});
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2};
        auto sm = aligner.align_hierarchy(t, altitudes);
        auto sm_k = graph_4_adjacency_2_khalimsky(g, {3, 3}, sm);

        array_1d<int> sm_coarse{2, 0, 0, 1, 1, 2, 0, 0, 0, 0, 2, 0};
        auto sm2 = aligner.align_hierarchy(g, sm_coarse);
        auto sm2_k = graph_4_adjacency_2_khalimsky(g, {3, 3}, sm2);

        array_2d<int> sm_k_ref = {{0, 2, 0, 1, 0},
                                  {0, 2, 1, 1, 0},
                                  {0, 2, 0, 0, 0},
                                  {0, 2, 0, 0, 0},
                                  {0, 2, 0, 0, 0}};

        REQUIRE((sm_k == sm_k_ref));
        REQUIRE((sm2_k == sm_k_ref));
    }

    TEST_CASE("hierarchy alignment 2", "[alignment]") {

        auto g = get_4_adjacency_graph({3, 3});
        array_1d<int> edge_weights{1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1};

        auto aligner = make_hierarchy_aligner_from_graph_cut(g, edge_weights);

        hg::tree t(array_1d<index_t>{9, 10, 10, 9, 11, 11, 9, 11, 11, 13, 12, 12, 13, 13});
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2};
        auto sm = aligner.align_hierarchy(t, altitudes);
        auto sm_k = graph_4_adjacency_2_khalimsky(g, {3, 3}, sm);

        array_1d<int> sm_coarse{2, 0, 0, 1, 1, 2, 0, 0, 0, 0, 2, 0};
        auto sm2 = aligner.align_hierarchy(g, sm_coarse);
        auto sm2_k = graph_4_adjacency_2_khalimsky(g, {3, 3}, sm2);

        array_2d<int> sm_k_ref = {{0, 2, 0, 1, 0},
                                  {0, 2, 1, 1, 0},
                                  {0, 2, 0, 0, 0},
                                  {0, 2, 0, 0, 0},
                                  {0, 2, 0, 0, 0}};

        REQUIRE((sm_k == sm_k_ref));
        REQUIRE((sm2_k == sm_k_ref));
    }

    TEST_CASE("hierarchy alignment with rag", "[alignment]") {

        auto g = get_4_adjacency_graph({3, 3});
        array_1d<int> edge_weights{1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1};

        auto aligner = make_hierarchy_aligner_from_graph_cut(g, edge_weights);


        hg::tree t(array_1d<index_t>{9, 10, 10, 9, 11, 11, 9, 11, 11, 13, 12, 12, 13, 13});
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2};

        auto res = supervertices_hierarchy(t, altitudes);
        auto &tree_res = res.tree;
        auto &supervertex_labelisation_res = res.supervertex_labelisation;
        auto &node_map_res = res.node_map;

        altitudes = hg::take(altitudes, node_map_res);

        auto sm = aligner.align_hierarchy(supervertex_labelisation_res, tree_res, altitudes);
        auto sm_k = graph_4_adjacency_2_khalimsky(g, {3, 3}, sm);

        array_2d<int> sm_k_ref = {{0, 2, 0, 1, 0},
                                  {0, 2, 1, 1, 0},
                                  {0, 2, 0, 0, 0},
                                  {0, 2, 0, 0, 0},
                                  {0, 2, 0, 0, 0}};

        REQUIRE((sm_k == sm_k_ref));
    }

    TEMPLATE_TEST_CASE("project fine to coarse labelisation empty unsigned labelisations", "[alignment]",
                       uint8_t, uint32_t, uint64_t) {
        array_1d<TestType> empty = hg::empty<TestType>({0});
        // max + 1 with the lowest value of the type on empty input: one region
        auto map = project_fine_to_coarse_labelisation(empty, empty);
        REQUIRE((map == array_1d<index_t>{0}));
        REQUIRE((project_fine_to_coarse_labelisation(empty, empty, 2, 3) == array_1d<index_t>{0, 0}));

        array_1d<TestType> fine_labels{0, 1, 2, 3, 4, 2, 3, 4, 2};
        array_1d<TestType> coarse_labels{0, 1, 1, 0, 2, 2, 0, 2, 2};
        REQUIRE((project_fine_to_coarse_labelisation(fine_labels, coarse_labels) ==
                 array_1d<index_t>{0, 1, 2, 0, 2}));
    }

    TEMPLATE_TEST_CASE("project fine to coarse labelisation empty signed labelisations", "[alignment]",
                       int8_t, int32_t, int64_t) {
        array_1d<TestType> empty = hg::empty<TestType>({0});
        array_1d<TestType> labels{0};
        REQUIRE_THROWS_WITH(project_fine_to_coarse_labelisation(empty, empty),
                            Catch::Matchers::ContainsSubstring("number of regions cannot be deduced"));
        REQUIRE_THROWS_AS(project_fine_to_coarse_labelisation(empty, empty, 2), std::runtime_error);
        REQUIRE_THROWS_AS(project_fine_to_coarse_labelisation(empty, empty, 0, 3), std::runtime_error);
        // explicit numbers of regions
        REQUIRE((project_fine_to_coarse_labelisation(empty, empty, 2, 3) == array_1d<index_t>{0, 0}));
        REQUIRE((project_fine_to_coarse_labelisation(labels, labels) == array_1d<index_t>{0}));
    }

    TEMPLATE_TEST_CASE("project fine to coarse labelisation invalid labels", "[alignment]",
                       int8_t, int32_t, int64_t) {
        using T = TestType;
        auto check_throws = [](const array_1d<T> &fine, const array_1d<T> &coarse,
                               size_t num_fine = 0, size_t num_coarse = 0) {
            REQUIRE_THROWS_AS(project_fine_to_coarse_labelisation(fine, coarse, num_fine, num_coarse),
                              std::runtime_error);
        };
        // negative labels, whatever the maximum
        check_throws({-1, -1}, {0, 0});
        check_throws({0, -1}, {0, 0});
        check_throws({0, 0}, {-1, -1});
        check_throws({0, 0}, {0, -1});
        check_throws({1, -5}, {0, 1});
        // labels not smaller than an explicit number of regions
        check_throws({0, 1}, {0, 0}, 1, 0);
        check_throws({0, 0}, {0, 1}, 0, 1);
        check_throws({0, 3}, {0, 1}, 2, 2);
        REQUIRE((project_fine_to_coarse_labelisation(array_1d<T>{0, 1}, array_1d<T>{1, 0}, 3, 2) ==
                 array_1d<index_t>{1, 0, 0}));
    }

    TEMPLATE_TEST_CASE("project fine to coarse labelisation labels above number of regions", "[alignment]",
                       uint8_t, uint64_t) {
        array_1d<TestType> fine{0, 1};
        array_1d<TestType> coarse{0, 0};
        REQUIRE_THROWS_AS(project_fine_to_coarse_labelisation(fine, coarse, 1), std::runtime_error);
        REQUIRE_THROWS_AS(project_fine_to_coarse_labelisation(coarse, fine, 0, 1), std::runtime_error);
        REQUIRE((project_fine_to_coarse_labelisation(fine, coarse, 2, 1) == array_1d<index_t>{0, 0}));
    }
}
