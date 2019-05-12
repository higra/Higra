/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/image/tree_of_shapes.hpp"
#include "higra/image/graph_image.hpp"
#include "higra/algo/tree.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xrandom.hpp"
#include "../test_utils.hpp"
#include <set>

namespace tree_of_shapes {

    using namespace hg;
    using namespace std;

    TEST_CASE("test integer_level_multi_queue", "[tree_of_shapes]") {
        using qt = hg::tree_of_shapes_internal::integer_level_multi_queue<int, int>;

        qt q(-2, 7);

        SECTION("empty queue") {
            REQUIRE(q.size() == 0);
            REQUIRE(q.empty());
            REQUIRE(q.num_levels() == 10);
            REQUIRE(q.min_level() == -2);
            REQUIRE(q.max_level() == 7);
            for (int i = -2; i < 8; i++) {
                REQUIRE(q.level_empty(i));
            }
        }
        SECTION("push top pop") {
            q.push(1, 10);
            REQUIRE(!q.level_empty(1));
            REQUIRE(q.size() == 1);
            q.push(1, 7);
            REQUIRE(q.size() == 2);
            REQUIRE(q.top(1) == 10);
            q.pop(1);
            REQUIRE(q.size() == 1);
            REQUIRE(q.top(1) == 7);
            q.pop(1);
            REQUIRE(q.size() == 0);
            REQUIRE(q.level_empty(1));
        }
        SECTION("closest non empty") {
            q.push(0, 4);
            q.push(5, 7);
            std::vector<int> res{0, 0, 0, 0, 0, 5, 5, 5, 5, 5};
            for (int i = -2; i < 8; i++) {
                REQUIRE(q.find_closest_non_empty_level(i) == res[i + 2]);
            }
        }
    }

    TEST_CASE("test interpolate_plain_map_khalimsky2d", "[tree_of_shapes]") {
        array_1d<int> image{1, 1, 1, 1, 1, 1,
                            1, 0, 0, 3, 3, 1,
                            1, 0, 1, 1, 3, 1,
                            1, 0, 0, 3, 3, 1,
                            1, 1, 1, 1, 1, 1};

        auto result = hg::tree_of_shapes_internal::interpolate_plain_map_khalimsky_2d(image, {5, 6});

        array_3d<int> expected_result
                {{{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 3}, {3, 3}, {3, 3}, {3, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 0}, {0, 1}, {0, 1}, {0, 3}, {1, 3}, {1, 3}, {3, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 0}, {0, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 3}, {3, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 0}, {0, 1}, {0, 1}, {0, 3}, {1, 3}, {1, 3}, {3, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 3}, {3, 3}, {3, 3}, {3, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}};

        REQUIRE((result == xt::reshape_view(expected_result, {result.shape()[0], result.shape()[1]})));
    }

    TEST_CASE("test sort_vertices_tree_of_shapes small integers", "[tree_of_shapes]") {
        array_nd<char> plain_map =
                {{{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 3}, {3, 3}, {3, 3}, {3, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 0}, {0, 1}, {0, 1}, {0, 3}, {1, 3}, {1, 3}, {3, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 0}, {0, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 3}, {3, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 0}, {0, 1}, {0, 1}, {0, 3}, {1, 3}, {1, 3}, {3, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 3}, {3, 3}, {3, 3}, {3, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}};
        plain_map.reshape({11 * 9, 2});

        auto g = get_4_adjacency_implicit_graph({9, 11});
        auto result = hg::tree_of_shapes_internal::sort_vertices_tree_of_shapes(g, plain_map, 1);
        auto &sorted_vertex_indices = result.first;
        auto &enqueued_level = result.second;

        set<index_t> zero_vertices{24, 25, 35, 26, 46, 57, 68, 69, 70};
        set<index_t> three_vertices{28, 29, 30, 41, 52, 72, 63, 73, 74};
        index_t num_ones = 9 * 11 - 2 * 9;
        set<index_t> zero_vertices_found{sorted_vertex_indices.begin() + num_ones,
                                         sorted_vertex_indices.begin() + num_ones + 9};
        REQUIRE(zero_vertices == zero_vertices_found);
        set<index_t> three_vertices_found{sorted_vertex_indices.begin() + num_ones + 9,
                                          sorted_vertex_indices.end()};
        REQUIRE(three_vertices == three_vertices_found);

        array_nd<char> expected_enqueued_level{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                               {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                               {1, 1, 0, 0, 0, 1, 3, 3, 3, 1, 1},
                                               {1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 1},
                                               {1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 1},
                                               {1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 1},
                                               {1, 1, 0, 0, 0, 1, 3, 3, 3, 1, 1},
                                               {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                               {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
        expected_enqueued_level.reshape({11 * 9});
        REQUIRE((enqueued_level == expected_enqueued_level));
    }

    TEST_CASE("test sort_vertices_tree_of_shapes float", "[tree_of_shapes]") {
        array_nd<float> plain_map =
                {{{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 3}, {3, 3}, {3, 3}, {3, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 0}, {0, 1}, {0, 1}, {0, 3}, {1, 3}, {1, 3}, {3, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 0}, {0, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 3}, {3, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 0}, {0, 1}, {0, 1}, {0, 3}, {1, 3}, {1, 3}, {3, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 3}, {3, 3}, {3, 3}, {3, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 1}},
                 {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}};
        plain_map.reshape({11 * 9, 2});

        auto g = get_4_adjacency_implicit_graph({9, 11});
        auto result = hg::tree_of_shapes_internal::sort_vertices_tree_of_shapes(g, plain_map, 1);
        auto &sorted_vertex_indices = result.first;
        auto &enqueued_level = result.second;

        set<index_t> zero_vertices{24, 25, 35, 26, 46, 57, 68, 69, 70};
        set<index_t> three_vertices{28, 29, 30, 41, 52, 72, 63, 73, 74};
        index_t num_ones = 9 * 11 - 2 * 9;
        set<index_t> zero_vertices_found{sorted_vertex_indices.begin() + num_ones,
                                         sorted_vertex_indices.begin() + num_ones + 9};
        REQUIRE(zero_vertices == zero_vertices_found);
        set<index_t> three_vertices_found{sorted_vertex_indices.begin() + num_ones + 9,
                                          sorted_vertex_indices.end()};
        REQUIRE(three_vertices == three_vertices_found);

        array_nd<float> expected_enqueued_level{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                                {1, 1, 0, 0, 0, 1, 3, 3, 3, 1, 1},
                                                {1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 1},
                                                {1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 1},
                                                {1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 1},
                                                {1, 1, 0, 0, 0, 1, 3, 3, 3, 1, 1},
                                                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
        expected_enqueued_level.reshape({11 * 9});
        REQUIRE((enqueued_level == expected_enqueued_level));
    }

    TEMPLATE_TEST_CASE("test tree of shapes no padding", "[tree_of_shapes]", char, float) {
        array_2d<TestType> image{{1, 1, 1, 1, 1, 1},
                                 {1, 0, 0, 3, 3, 1},
                                 {1, 0, 1, 1, 3, 1},
                                 {1, 0, 0, 3, 3, 1},
                                 {1, 1, 1, 1, 1, 1}};

        auto result = component_tree_tree_of_shapes_image2d(image, tos_padding::none, false);
        auto &tree = result.tree;
        auto &altitudes = result.altitudes;
        array_1d<index_t>
                ref_parents{101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
                            101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
                            101, 101, 100, 100, 100, 101, 99, 99, 99, 101, 101,
                            101, 101, 100, 101, 101, 101, 101, 101, 99, 101, 101,
                            101, 101, 100, 101, 101, 101, 101, 101, 99, 101, 101,
                            101, 101, 100, 101, 101, 101, 101, 101, 99, 101, 101,
                            101, 101, 100, 100, 100, 101, 99, 99, 99, 101, 101,
                            101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
                            101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
                            101, 101, 101};
        array_1d<TestType>
                ref_altitudes{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 0, 0, 0, 1, 3, 3, 3, 1, 1,
                              1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 1,
                              1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 1,
                              1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 1,
                              1, 1, 0, 0, 0, 1, 3, 3, 3, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              3, 0, 1};
        REQUIRE((tree.parents() == ref_parents));
        REQUIRE((altitudes == ref_altitudes));
    }

    TEMPLATE_TEST_CASE("test tree of shapes no padding original space", "[tree_of_shapes]", char, float) {
        array_2d<TestType> image{{1, 1, 1, 1, 1, 1},
                                 {1, 0, 0, 3, 3, 1},
                                 {1, 0, 1, 1, 3, 1},
                                 {1, 0, 0, 3, 3, 1},
                                 {1, 1, 1, 1, 1, 1}};

        auto result = component_tree_tree_of_shapes_image2d(image, tos_padding::none, true);
        auto &tree = result.tree;
        auto &altitudes = result.altitudes;

        array_1d<index_t>
                ref_parents{32, 32, 32, 32, 32, 32,
                            32, 31, 31, 30, 30, 32,
                            32, 31, 32, 32, 30, 32,
                            32, 31, 31, 30, 30, 32,
                            32, 32, 32, 32, 32, 32,
                            32, 32, 32};
        array_1d<TestType>
                ref_altitudes{1, 1, 1, 1, 1, 1,
                              1, 0, 0, 3, 3, 1,
                              1, 0, 1, 1, 3, 1,
                              1, 0, 0, 3, 3, 1,
                              1, 1, 1, 1, 1, 1,
                              3, 0, 1};
        REQUIRE((tree.parents() == ref_parents));
        REQUIRE((altitudes == ref_altitudes));
    }

    TEST_CASE("test tree of shapes padding 0", "[tree_of_shapes]") {
        array_2d<float> image{{1, 1,  1},
                              {1, -2, 3}};

        auto result = component_tree_tree_of_shapes_image2d(image, tos_padding::zero, false);
        auto &tree = result.tree;
        auto &altitudes = result.altitudes;

        array_1d<index_t>
                ref_parents{66, 66, 66, 66, 66, 66, 66, 66, 66,
                            66, 66, 66, 66, 66, 66, 66, 66, 66,
                            66, 66, 65, 65, 65, 65, 65, 66, 66,
                            66, 66, 65, 66, 66, 66, 65, 66, 66,
                            66, 66, 65, 66, 63, 66, 64, 66, 66,
                            66, 66, 66, 66, 66, 66, 66, 66, 66,
                            66, 66, 66, 66, 66, 66, 66, 66, 66,
                            66, 65, 66, 66};
        array_1d<float>
                ref_altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 1, 1, 1, 1, 1, 0, 0,
                              0, 0, 1, 0, 0, 0, 1, 0, 0,
                              0, 0, 1, 0, -2, 0, 3, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0,
                              -2, 3, 1, 0};
        REQUIRE((tree.parents() == ref_parents));
        REQUIRE((altitudes == ref_altitudes));
    }

    TEST_CASE("test tree of shapes padding 0 original space", "[tree_of_shapes]") {
        array_2d<float> image{{1, 1,  1},
                              {1, -2, 3}};

        auto result = component_tree_tree_of_shapes_image2d(image, tos_padding::zero, true);
        auto &tree = result.tree;
        auto &altitudes = result.altitudes;

        array_1d<index_t>
                ref_parents{8, 8, 8,
                            8, 6, 7,
                            9, 8, 9, 9};
        array_1d<char>
                ref_altitudes{1, 1, 1,
                              1, -2, 3,
                              -2, 3, 1, 0};
        REQUIRE((tree.parents() == ref_parents));
        REQUIRE((altitudes == ref_altitudes));
    }

    TEST_CASE("test tree of shapes padding mean original space", "[tree_of_shapes]") {
        array_2d<float> image{{1, 1},
                              {1, -2},
                              {1, 7}};

        auto result = component_tree_tree_of_shapes_image2d(image, tos_padding::mean, true);
        auto &tree = result.tree;
        auto &altitudes = result.altitudes;

        array_1d<index_t>
                ref_parents{8, 8,
                            8, 7,
                            8, 6,
                            9, 8, 9, 9};
        array_1d<float>
                ref_altitudes{1., 1.,
                              1., -2.,
                              1., 7.,
                              7., -2., 1., 1.5};
        REQUIRE((tree.parents() == ref_parents));
        REQUIRE((altitudes == ref_altitudes));
    }

    TEST_CASE("test tree of shapes self duality", "[tree_of_shapes]") {
        xt::random::seed(42);
        array_2d<double> image = xt::random::rand<double>({25, 38});
        auto res1 = component_tree_tree_of_shapes_image2d(image);
        auto res2 = component_tree_tree_of_shapes_image2d(-image);
        REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));
    }

}
