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
#include "../test_utils.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
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
        }SECTION("push top pop") {
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
        }SECTION("closest non empty") {
            q.push(0, 4);
            q.push(5, 7);
            std::vector<int> res{0, 0, 0, 0, 0, 5, 5, 5, 5, 5};
            for (int i = -2; i < 8; i++) {
                REQUIRE(q.find_closest_non_empty_level(i) == res[i + 2]);
            }
        }
    }
    /*
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

        REQUIRE((result == expected_result.reshaped({result.shape()[0], result.shape()[1]})));
    }*/

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

TEMPLATE_TEST_CASE("test tree of shapes 2D no padding", "[tree_of_shapes]", char, float) {
    array_2d <TestType> image{{1, 1, 1, 1, 1, 1},
                              {1, 0, 0, 3, 3, 1},
                              {1, 0, 1, 1, 3, 1},
                              {1, 0, 0, 3, 3, 1},
                              {1, 1, 1, 1, 1, 1}};

    auto result = component_tree_tree_of_shapes_image2d(image, tos_padding::none, false);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;
    array_1d <index_t>
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
    array_1d <TestType>
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

TEMPLATE_TEST_CASE("test tree of shapes 2D no padding original space", "[tree_of_shapes]", char, float) {
    array_2d <TestType> image{{1, 1, 1, 1, 1, 1},
                              {1, 0, 0, 3, 3, 1},
                              {1, 0, 1, 1, 3, 1},
                              {1, 0, 0, 3, 3, 1},
                              {1, 1, 1, 1, 1, 1}};

    auto result = component_tree_tree_of_shapes_image2d(image, tos_padding::none, true);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{32, 32, 32, 32, 32, 32,
                        32, 30, 30, 31, 31, 32,
                        32, 30, 32, 32, 31, 32,
                        32, 30, 30, 31, 31, 32,
                        32, 32, 32, 32, 32, 32,
                        32, 32, 32};
    array_1d <TestType>
            ref_altitudes{1, 1, 1, 1, 1, 1,
                          1, 0, 0, 3, 3, 1,
                          1, 0, 1, 1, 3, 1,
                          1, 0, 0, 3, 3, 1,
                          1, 1, 1, 1, 1, 1,
                          0, 3, 1};
    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

TEST_CASE("test tree of shapes 2D padding 0", "[tree_of_shapes]") {
    array_2d<float> image{{1, 1,  1},
                          {1, -2, 3}};

    auto result = component_tree_tree_of_shapes_image2d(image, tos_padding::zero, false);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
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

TEST_CASE("test tree of shapes 2D padding 0 original space", "[tree_of_shapes]") {
    array_2d<float> image{{1, 1,  1},
                          {1, -2, 3}};

    auto result = component_tree_tree_of_shapes_image2d(image, tos_padding::zero, true);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{7, 7, 7,
                        7, 8, 6,
                        7, 9, 9, 9};
    array_1d<char>
            ref_altitudes{1, 1, 1,
                          1, -2, 3,
                          3, 1, -2, 0};
    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

TEST_CASE("test tree of shapes 2D padding mean original space", "[tree_of_shapes]") {
    array_2d<float> image{{1, 1},
                          {1, -2},
                          {1, 7}};

    auto result = component_tree_tree_of_shapes_image2d(image, tos_padding::mean, true);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{7, 7,
                        7, 6,
                        7, 8,
                        7, 9, 9, 9};
    array_1d<float>
            ref_altitudes{1., 1.,
                          1., -2.,
                          1., 7.,
                          -2., 1., 7., 1.5};
    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

TEST_CASE("test tree of shapes 2D no immersion no padding original space", "[tree_of_shapes]") {
    array_nd<float> image{{1, 1, 1, 1, 1},
                          {1, 0, 1, 2, 1},
                          {1, 1, 1, 1, 1}};

    auto result = component_tree_tree_of_shapes_image2d(image, tos_padding::none, /*original*/true, /*immersion*/false);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{17, 17, 17, 17, 17,
                        17, 16, 17, 15, 17,
                        17, 17, 17, 17, 17, 17, 17, 17};
    array_1d<float> ref_altitudes{1, 1, 1, 1, 1,
                                  1, 0, 1, 2, 1,
                                  1, 1, 1, 1, 1, 2, 0, 1};

    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

TEST_CASE("test tree of shapes 2D no immersion padding zero original space", "[tree_of_shapes]") {
    array_nd<float> image{{1, 1, 1, 1, 1},
                          {1, 0, 1, 2, 1},
                          {1, 1, 1, 1, 1}};

    auto result = component_tree_tree_of_shapes_image2d(image, tos_padding::zero, /*original*/true, /*immersion*/false);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{17, 17, 17, 17, 17,
                        17, 15, 17, 16, 17,
                        17, 17, 17, 17, 17, 17, 17, 18, 18};
    array_1d<float> ref_altitudes{1, 1, 1, 1, 1,
                                  1, 0, 1, 2, 1,
                                  1, 1, 1, 1, 1, 0, 2, 1, 0};

    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

TEST_CASE("test tree of shapes 2D no immersion no padding no original space", "[tree_of_shapes]") {
    array_nd<float> image{{1, 1, 1, 1, 1},
                          {1, 0, 1, 2, 1},
                          {1, 1, 1, 1, 1}};

    auto result = component_tree_tree_of_shapes_image2d(image, tos_padding::none, /*original*/false, /*immersion*/
                                                        false);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{17, 17, 17, 17, 17,
                        17, 16, 17, 15, 17,
                        17, 17, 17, 17, 17, 17, 17, 17};
    array_1d<float> ref_altitudes{1, 1, 1, 1, 1,
                                  1, 0, 1, 2, 1,
                                  1, 1, 1, 1, 1, 2, 0, 1};
    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

TEST_CASE("test tree of shapes 2D no immersion padding zero no original space", "[tree_of_shapes]") {
    array_nd<float> image{{1, 1, 1, 1, 1},
                          {1, 0, 1, 2, 1},
                          {1, 1, 1, 1, 1}};

    auto result = component_tree_tree_of_shapes_image2d(image, tos_padding::zero, /*original*/false, /*immersion*/
                                                        false);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{38, 38, 38, 38, 38, 38, 38,
                        38, 37, 37, 37, 37, 37, 38,
                        38, 37, 36, 37, 35, 37, 38,
                        38, 37, 37, 37, 37, 37, 38,
                        38, 38, 38, 38, 38, 38, 38, 37, 37, 38, 38};
    array_1d<float> ref_altitudes{0, 0, 0, 0, 0, 0, 0,
                                  0, 1, 1, 1, 1, 1, 0,
                                  0, 1, 0, 1, 2, 1, 0,
                                  0, 1, 1, 1, 1, 1, 0,
                                  0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0};

    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}


TEST_CASE("test tree of shapes 2D self duality", "[tree_of_shapes]") {
    hg_test::random_seed(42);
    array_2d<double> image = hg_test::random_uniform<double>({25, 38});
    auto res1 = component_tree_tree_of_shapes_image2d(image);
    auto res2 = component_tree_tree_of_shapes_image2d(hg_test::negated(image));
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));
}

// 3D ToS tests

TEST_CASE("test tree of shapes 3D self duality", "[tree_of_shapes]") {
    hg_test::random_seed(42);
    array_3d<double> image = hg_test::random_uniform<double>({8, 15, 12});

    auto res1 = component_tree_tree_of_shapes_image3d(image, tos_padding::mean, true, true);
    auto res2 = component_tree_tree_of_shapes_image3d(hg_test::negated(image),tos_padding::mean, true, true);
    
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));

    res1 = component_tree_tree_of_shapes_image3d(image, tos_padding::mean, true, false);
    res2 = component_tree_tree_of_shapes_image3d(hg_test::negated(image),tos_padding::mean, true, false);
    
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));

    res1 = component_tree_tree_of_shapes_image3d(image, tos_padding::mean, false, false);
    res2 = component_tree_tree_of_shapes_image3d(hg_test::negated(image),tos_padding::mean, false, false);
    
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));

    res1 = component_tree_tree_of_shapes_image3d(image, tos_padding::mean, false, true);
    res2 = component_tree_tree_of_shapes_image3d(hg_test::negated(image),tos_padding::mean, false, true);
    
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));

    res1 = component_tree_tree_of_shapes_image3d(image, tos_padding::none, false, false);
    res2 = component_tree_tree_of_shapes_image3d(hg_test::negated(image),tos_padding::none, false, false);
    
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));

    res1 = component_tree_tree_of_shapes_image3d(image, tos_padding::none, true, false);
    res2 = component_tree_tree_of_shapes_image3d(hg_test::negated(image),tos_padding::none, true, false);
    
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));

    res1 = component_tree_tree_of_shapes_image3d(image, tos_padding::none, true, true);
    res2 = component_tree_tree_of_shapes_image3d(hg_test::negated(image),tos_padding::none, true, true);
    
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));

    res1 = component_tree_tree_of_shapes_image3d(image, tos_padding::none, false, true);
    res2 = component_tree_tree_of_shapes_image3d(hg_test::negated(image),tos_padding::none, false, true);
    
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));
}

TEST_CASE("test tree of shapes 3D flat no immersion no padding original space", "[tree_of_shapes]") {
    // A "3D" array which is the same as in previous unit test should 
    // give the same result
    array_nd<float> image{{{1, 1, 1, 1, 1},
                          {1, 0, 1, 2, 1},
                          {1, 1, 1, 1, 1}}};

    auto result = component_tree_tree_of_shapes_image3d(image, tos_padding::none, /*original*/true, /*immersion*/false);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{17, 17, 17, 17, 17,
                        17, 16, 17, 15, 17,
                        17, 17, 17, 17, 17, 17, 17, 17};
    array_1d<float> ref_altitudes{1, 1, 1, 1, 1,
                                  1, 0, 1, 2, 1,
                                  1, 1, 1, 1, 1, 2, 0, 1};

    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

TEST_CASE("test tree of shapes 3D flat default param", "[tree_of_shapes]") {

    array_nd<float> image{{{1, 1},
                          {1, -2},
                          {1, 7}}};

    auto result = component_tree_tree_of_shapes_image3d(image, tos_padding::mean, true);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{7, 7,
                        7, 6,
                        7, 8,
                        7, 9, 9, 9};
    array_1d<float>
            ref_altitudes{1., 1.,
                          1., -2.,
                          1., 7.,
                          -2., 1., 7., 1.5};


    INFO("altitudes are : " << altitudes << "\nexpected : " << ref_altitudes);
    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

TEST_CASE("test tree of shapes 3D default param", "[tree_of_shapes]") {

    array_nd<float> image = {{{1, 1, 1},
                             {1, 1, 1},
                             {1, 1, 1}},
                             {{1, 1, 1},
                             {1, -2, 1},
                             {1, 1, 1}},
                             {{1, 1, 1},
                             {1, 1, 1},
                             {1, 1, 1}}};

    auto result = component_tree_tree_of_shapes_image3d(image, tos_padding::mean, true);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d<float>
            ref_parents{28., 28., 28.,
                        28., 28., 28.,
                        28., 28., 28.,
                        28., 28., 28.,
                        28., 27., 28.,
                        28., 28., 28.,
                        28., 28., 28.,
                        28., 28., 28.,
                        28., 28., 28., 28., 28};
    array_1d<float>
            ref_altitudes{1., 1. , 1.,
                          1., 1. , 1.,
                          1., 1. , 1.,
                          1., 1. , 1.,
                          1., -2., 1.,
                          1., 1. , 1.,
                          1., 1. , 1.,
                          1., 1. , 1.,
                          1., 1. , 1., -2., 1.};


    INFO("altitudes are : " << altitudes << "\nexpected : " << ref_altitudes);
    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

//  generic ToS tests

TEST_CASE("test tree of shapes self duality", "[tree_of_shapes]") {
    hg_test::random_seed(42);
    array_3d<double> image = hg_test::random_uniform<double>({25, 38, 24});

    auto res1 = component_tree_tree_of_shapes_image(image, tos_padding::mean, true, true);
    auto res2 = component_tree_tree_of_shapes_image(hg_test::negated(image),tos_padding::mean, true, true);
    
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));

    res1 = component_tree_tree_of_shapes_image(image, tos_padding::mean, true, false);
    res2 = component_tree_tree_of_shapes_image(hg_test::negated(image),tos_padding::mean, true, false);
    
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));

    res1 = component_tree_tree_of_shapes_image(image, tos_padding::mean, false, false);
    res2 = component_tree_tree_of_shapes_image(hg_test::negated(image),tos_padding::mean, false, false);
    
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));

    res1 = component_tree_tree_of_shapes_image(image, tos_padding::mean, false, true);
    res2 = component_tree_tree_of_shapes_image(hg_test::negated(image),tos_padding::mean, false, true);
    
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));

    res1 = component_tree_tree_of_shapes_image(image, tos_padding::none, false, false);
    res2 = component_tree_tree_of_shapes_image(hg_test::negated(image),tos_padding::none, false, false);
    
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));

    res1 = component_tree_tree_of_shapes_image(image, tos_padding::none, true, false);
    res2 = component_tree_tree_of_shapes_image(hg_test::negated(image),tos_padding::none, true, false);
    
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));

    res1 = component_tree_tree_of_shapes_image(image, tos_padding::none, true, true);
    res2 = component_tree_tree_of_shapes_image(hg_test::negated(image),tos_padding::none, true, true);
    
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));

    res1 = component_tree_tree_of_shapes_image(image, tos_padding::none, false, true);
    res2 = component_tree_tree_of_shapes_image(hg_test::negated(image),tos_padding::none, false, true);
    
    REQUIRE(test_tree_isomorphism(res1.tree, res2.tree));
}

TEST_CASE("test tree of shapes flat no immersion no padding original space", "[tree_of_shapes]") {
    // A "3D" array which is the same as in previous unit test should 
    // give the same result
    array_nd<float> image{{{1, 1, 1, 1, 1},
                          {1, 0, 1, 2, 1},
                          {1, 1, 1, 1, 1}}};

    auto result = component_tree_tree_of_shapes_image(image, tos_padding::none, /*original*/true, /*immersion*/false);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{17, 17, 17, 17, 17,
                        17, 16, 17, 15, 17,
                        17, 17, 17, 17, 17, 17, 17, 17};
    array_1d<float> ref_altitudes{1, 1, 1, 1, 1,
                                  1, 0, 1, 2, 1,
                                  1, 1, 1, 1, 1, 2, 0, 1};

    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

TEST_CASE("test tree of shapes default param 1", "[tree_of_shapes]") {

    array_nd<float> image{{{1, 1},
                          {1, -2},
                          {1, 7}}};

    auto result = component_tree_tree_of_shapes_image(image, tos_padding::mean, true);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{7, 7,
                        7, 6,
                        7, 8,
                        7, 9, 9, 9};
    array_1d<float>
            ref_altitudes{1., 1.,
                          1., -2.,
                          1., 7.,
                          -2., 1., 7., 1.5};

    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

TEST_CASE("test tree of shapes default param 2", "[tree_of_shapes]") {

    array_nd<float> image = {{{1, 1, 1},
                             {1, 1, 1},
                             {1, 1, 1}},
                             {{1, 1, 1},
                             {1, -2, 1},
                             {1, 1, 1}},
                             {{1, 1, 1},
                             {1, 1, 1},
                             {1, 1, 1}}};

    auto result = component_tree_tree_of_shapes_image(image, tos_padding::mean, true);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d<float>
            ref_parents{28., 28., 28.,
                        28., 28., 28.,
                        28., 28., 28.,
                        28., 28., 28.,
                        28., 27., 28.,
                        28., 28., 28.,
                        28., 28., 28.,
                        28., 28., 28.,
                        28., 28., 28., 28., 28};
    array_1d<float>
            ref_altitudes{1., 1. , 1.,
                          1., 1. , 1.,
                          1., 1. , 1.,
                          1., 1. , 1.,
                          1., -2., 1.,
                          1., 1. , 1.,
                          1., 1. , 1.,
                          1., 1. , 1.,
                          1., 1. , 1., -2., 1.};


    INFO("altitudes are : " << altitudes << "\nexpected : " << ref_altitudes);
    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}


TEMPLATE_TEST_CASE("test tree of shapes no padding", "[tree_of_shapes]", char, float) {
    array_2d <TestType> image{{1, 1, 1, 1, 1, 1},
                              {1, 0, 0, 3, 3, 1},
                              {1, 0, 1, 1, 3, 1},
                              {1, 0, 0, 3, 3, 1},
                              {1, 1, 1, 1, 1, 1}};

    auto result = component_tree_tree_of_shapes_image(image, tos_padding::none, false);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;
    array_1d <index_t>
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
    array_1d <TestType>
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
    array_2d <TestType> image{{1, 1, 1, 1, 1, 1},
                              {1, 0, 0, 3, 3, 1},
                              {1, 0, 1, 1, 3, 1},
                              {1, 0, 0, 3, 3, 1},
                              {1, 1, 1, 1, 1, 1}};

    auto result = component_tree_tree_of_shapes_image(image, tos_padding::none, true);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{32, 32, 32, 32, 32, 32,
                        32, 30, 30, 31, 31, 32,
                        32, 30, 32, 32, 31, 32,
                        32, 30, 30, 31, 31, 32,
                        32, 32, 32, 32, 32, 32,
                        32, 32, 32};
    array_1d <TestType>
            ref_altitudes{1, 1, 1, 1, 1, 1,
                          1, 0, 0, 3, 3, 1,
                          1, 0, 1, 1, 3, 1,
                          1, 0, 0, 3, 3, 1,
                          1, 1, 1, 1, 1, 1,
                          0, 3, 1};
    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

TEST_CASE("test tree of shapes padding 0", "[tree_of_shapes]") {
    array_2d<float> image{{1, 1,  1},
                          {1, -2, 3}};

    auto result = component_tree_tree_of_shapes_image(image, tos_padding::zero, false);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
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

    auto result = component_tree_tree_of_shapes_image(image, tos_padding::zero, true);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{7, 7, 7,
                        7, 8, 6,
                        7, 9, 9, 9};
    array_1d<char>
            ref_altitudes{1, 1, 1,
                          1, -2, 3,
                          3, 1, -2, 0};
    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

TEST_CASE("test tree of shapes padding mean original space", "[tree_of_shapes]") {
    array_2d<float> image{{1, 1},
                          {1, -2},
                          {1, 7}};

    auto result = component_tree_tree_of_shapes_image(image, tos_padding::mean, true);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{7, 7,
                        7, 6,
                        7, 8,
                        7, 9, 9, 9};
    array_1d<float>
            ref_altitudes{1., 1.,
                          1., -2.,
                          1., 7.,
                          -2., 1., 7., 1.5};
    INFO("altitudes are : " << altitudes << "\nexpected : " << ref_altitudes);

    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

TEST_CASE("test tree of shapes no immersion no padding original space", "[tree_of_shapes]") {
    array_nd<float> image{{1, 1, 1, 1, 1},
                          {1, 0, 1, 2, 1},
                          {1, 1, 1, 1, 1}};

    auto result = component_tree_tree_of_shapes_image(image, tos_padding::none, /*original*/true, /*immersion*/false);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{17, 17, 17, 17, 17,
                        17, 16, 17, 15, 17,
                        17, 17, 17, 17, 17, 17, 17, 17};
    array_1d<float> ref_altitudes{1, 1, 1, 1, 1,
                                  1, 0, 1, 2, 1,
                                  1, 1, 1, 1, 1, 2, 0, 1};

    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

TEST_CASE("test tree of shapes no immersion padding zero original space", "[tree_of_shapes]") {
    array_nd<float> image{{1, 1, 1, 1, 1},
                          {1, 0, 1, 2, 1},
                          {1, 1, 1, 1, 1}};

    auto result = component_tree_tree_of_shapes_image(image, tos_padding::zero, /*original*/true, /*immersion*/false);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{17, 17, 17, 17, 17,
                        17, 15, 17, 16, 17,
                        17, 17, 17, 17, 17, 17, 17, 18, 18};
    array_1d<float> ref_altitudes{1, 1, 1, 1, 1,
                                  1, 0, 1, 2, 1,
                                  1, 1, 1, 1, 1, 0, 2, 1, 0};

    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

TEST_CASE("test tree of shapes no immersion no padding no original space", "[tree_of_shapes]") {
    array_nd<float> image{{1, 1, 1, 1, 1},
                          {1, 0, 1, 2, 1},
                          {1, 1, 1, 1, 1}};

    auto result = component_tree_tree_of_shapes_image(image, tos_padding::none, /*original*/false, /*immersion*/
                                                        false);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{17, 17, 17, 17, 17,
                        17, 16, 17, 15, 17,
                        17, 17, 17, 17, 17, 17, 17, 17};
    array_1d<float> ref_altitudes{1, 1, 1, 1, 1,
                                  1, 0, 1, 2, 1,
                                  1, 1, 1, 1, 1, 2, 0, 1};
    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}

TEST_CASE("test tree of shapes no immersion padding zero no original space", "[tree_of_shapes]") {
    array_nd<float> image{{1, 1, 1, 1, 1},
                          {1, 0, 1, 2, 1},
                          {1, 1, 1, 1, 1}};

    auto result = component_tree_tree_of_shapes_image(image, tos_padding::zero, /*original*/false, /*immersion*/
                                                        false);
    auto &tree = result.tree;
    auto &altitudes = result.altitudes;

    array_1d <index_t>
            ref_parents{38, 38, 38, 38, 38, 38, 38,
                        38, 37, 37, 37, 37, 37, 38,
                        38, 37, 36, 37, 35, 37, 38,
                        38, 37, 37, 37, 37, 37, 38,
                        38, 38, 38, 38, 38, 38, 38, 37, 37, 38, 38};
    array_1d<float> ref_altitudes{0, 0, 0, 0, 0, 0, 0,
                                  0, 1, 1, 1, 1, 1, 0,
                                  0, 1, 0, 1, 2, 1, 0,
                                  0, 1, 1, 1, 1, 1, 0,
                                  0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0};

    REQUIRE((tree.parents() == ref_parents));
    REQUIRE((altitudes == ref_altitudes));
}


TEST_CASE("test tree of shapes exterior vertex range", "[tree_of_shapes]") {
    array_2d<float> image2d{{0, 3, 1, 4},
                            {2, 0, 4, 1},
                            {1, 4, 0, 2}};
    array_3d<float> image3d = array_3d<float>::from_shape({2, 3, 4});
    for (index_t i = 0; i < (index_t) image3d.size(); i++) {
        image3d.data()[i] = (float) (i % 5);
    }

    auto check = [](const auto &image) {
        const auto shape = image.shape();
        for (auto padding: {tos_padding::none, tos_padding::zero, tos_padding::mean}) {
            for (bool immersion: {true, false}) {
                for (bool original_size: {true, false}) {
                    // number of vertices of the padded/interpolated space
                    index_t num_vertices = 1;
                    for (index_t i = 0; i < (index_t) image.dimension(); i++) {
                        index_t s = (index_t) shape[i] + ((padding == tos_padding::none) ? 0 : 2);
                        num_vertices *= (immersion) ? s * 2 - 1 : s;
                    }
                    CAPTURE(image.dimension(), (int) padding, immersion, original_size, num_vertices);
                    for (index_t exterior_vertex: {(index_t) 0, num_vertices - 1}) {
                        REQUIRE_NOTHROW(component_tree_tree_of_shapes_image(image, padding, original_size,
                                                                           immersion, exterior_vertex));
                    }
                    for (index_t exterior_vertex: {(index_t) -1, num_vertices, num_vertices * 10}) {
                        REQUIRE_THROWS_AS(component_tree_tree_of_shapes_image(image, padding, original_size,
                                                                             immersion, exterior_vertex),
                                          std::runtime_error);
                    }
                }
            }
        }
    };
    check(image2d);
    check(image3d);

    REQUIRE_THROWS_AS(component_tree_tree_of_shapes_image2d(image2d, tos_padding::mean, true, true, 99),
                      std::runtime_error);
    REQUIRE_NOTHROW(component_tree_tree_of_shapes_image2d(image2d, tos_padding::mean, true, true, 98));
    REQUIRE_THROWS_AS(component_tree_tree_of_shapes_image3d(image3d, tos_padding::mean, true, true, 693),
                      std::runtime_error);
    REQUIRE_NOTHROW(component_tree_tree_of_shapes_image3d(image3d, tos_padding::mean, true, true, 692));
}

TEMPLATE_TEST_CASE("test tree of shapes empty image", "[tree_of_shapes]", unsigned char, int, float) {
    const std::vector<array_shape> shapes{{0, 4}, {3, 0}, {0, 0}, {0, 3, 4}, {2, 0, 4}, {2, 3, 0}};
    for (const auto &shape: shapes) {
        hg::array<TestType> image = hg::array<TestType>::from_shape(shape);
        for (auto padding: {tos_padding::none, tos_padding::zero, tos_padding::mean}) {
            for (bool immersion: {true, false}) {
                for (bool original_size: {true, false}) {
                    CAPTURE(shape.size(), shape[0], shape[1], (int) padding, immersion, original_size);
                    REQUIRE_THROWS_AS(component_tree_tree_of_shapes_image(image, padding, original_size, immersion),
                                      std::runtime_error);
                }
            }
        }
    }

    array_2d<TestType> image2d = array_2d<TestType>::from_shape({0, 4});
    REQUIRE_THROWS_AS(component_tree_tree_of_shapes_image2d(image2d), std::runtime_error);
    array_3d<TestType> image3d = array_3d<TestType>::from_shape({2, 0, 4});
    REQUIRE_THROWS_AS(component_tree_tree_of_shapes_image3d(image3d), std::runtime_error);
}

TEST_CASE("test tree of shapes interval midpoint", "[tree_of_shapes]") {
    using tree_of_shapes_internal::interval_midpoint;
    using i64 = std::int64_t;
    using i32 = std::int32_t;
    using u64 = std::uint64_t;
    const i64 i64_max = std::numeric_limits<i64>::max();
    const i64 i64_min = std::numeric_limits<i64>::min();
    const i32 i32_max = std::numeric_limits<i32>::max();
    const i32 i32_min = std::numeric_limits<i32>::min();

    // equal endpoints
    REQUIRE(interval_midpoint<i64>((i64(1) << 53) + 1, (i64(1) << 53) + 1) == (i64(1) << 53) + 1);
    REQUIRE(interval_midpoint<i64>(i64_max, i64_max) == i64_max);
    REQUIRE(interval_midpoint<i64>(i64_min, i64_min) == i64_min);
    REQUIRE(interval_midpoint<i32>(i32(1) << 30, i32(1) << 30) == (i32(1) << 30));

    // truncation toward zero, in both argument orders
    REQUIRE(interval_midpoint<i32>(-3, 0) == -1);
    REQUIRE(interval_midpoint<i32>(0, -3) == -1);
    REQUIRE(interval_midpoint<i32>(-2, 1) == 0);
    REQUIRE(interval_midpoint<i32>(-1, 2) == 0);
    REQUIRE(interval_midpoint<i32>(1, 4) == 2);
    REQUIRE(interval_midpoint<i32>(-4, -1) == -2);
    REQUIRE(interval_midpoint<signed char>(-128, 127) == 0);
    REQUIRE(interval_midpoint<signed char>(-128, -127) == -127);
    REQUIRE(interval_midpoint<unsigned char>(255, 254) == 254);

    // extreme values
    REQUIRE(interval_midpoint<i32>(i32_min, i32_max) == 0);
    REQUIRE(interval_midpoint<i32>(i32_max - 2, i32_max) == i32_max - 1);
    REQUIRE(interval_midpoint<i32>(i32_min, i32_min + 2) == i32_min + 1);
    REQUIRE(interval_midpoint<i64>(i64_min, i64_max) == 0);
    REQUIRE(interval_midpoint<i64>(i64_max - 1, i64_max) == i64_max - 1);
    REQUIRE(interval_midpoint<i64>(i64_min, i64_min + 1) == i64_min + 1);
    REQUIRE(interval_midpoint<u64>(std::numeric_limits<u64>::max(), std::numeric_limits<u64>::max() - 2) ==
            std::numeric_limits<u64>::max() - 1);

    // floating point
    const float f_max = std::numeric_limits<float>::max();
    const float f_denorm = std::numeric_limits<float>::denorm_min();
    REQUIRE(interval_midpoint<float>(2e38f, 2e38f) == 2e38f);
    REQUIRE(interval_midpoint<float>(f_max, f_max) == f_max);
    REQUIRE(interval_midpoint<float>(f_max, f_max / 2) == 0.75f * f_max);
    REQUIRE(interval_midpoint<float>(-f_max, f_max) == 0);
    REQUIRE(interval_midpoint<float>(1, 2) == 1.5f);
    REQUIRE(interval_midpoint<float>(-1.5f, 0) == -0.75f);
    REQUIRE(interval_midpoint<float>(f_denorm, 3 * f_denorm) == 2 * f_denorm);
    REQUIRE(interval_midpoint<double>(std::numeric_limits<double>::max(), std::numeric_limits<double>::max()) ==
            std::numeric_limits<double>::max());
}

TEMPLATE_TEST_CASE("test tree of shapes root altitude at exterior pixel", "[tree_of_shapes]",
                   std::int32_t, std::int64_t, float) {
    TestType value;
    if constexpr (std::is_same<TestType, std::int64_t>::value) {
        value = ((std::int64_t) 1 << 53) + 1;
    } else if constexpr (std::is_same<TestType, std::int32_t>::value) {
        value = (std::int32_t) 1 << 30;
    } else {
        value = 2e38f;
    }

    array_2d<TestType> image2d = array_2d<TestType>::from_shape({3, 3});
    std::fill(image2d.begin(), image2d.end(), (TestType) 0);
    image2d(1, 1) = value;
    array_3d<TestType> image3d = array_3d<TestType>::from_shape({3, 3, 3});
    std::fill(image3d.begin(), image3d.end(), (TestType) 0);
    image3d(1, 1, 1) = value;

    for (bool immersion: {false, true}) {
        CAPTURE(immersion);
        // centre pixel in the working space without padding
        index_t exterior2d = (immersion) ? 2 * 5 + 2 : 4;
        index_t exterior3d = (immersion) ? (2 * 5 + 2) * 5 + 2 : 13;
        auto res2d = component_tree_tree_of_shapes_image(image2d, tos_padding::none, true, immersion, exterior2d);
        REQUIRE(res2d.altitudes(root(res2d.tree)) == value);
        auto res3d = component_tree_tree_of_shapes_image(image3d, tos_padding::none, true, immersion, exterior3d);
        REQUIRE(res3d.altitudes(root(res3d.tree)) == value);
        auto res2d_wrapper = component_tree_tree_of_shapes_image2d(image2d, tos_padding::none, true, immersion,
                                                                   exterior2d);
        REQUIRE(res2d_wrapper.altitudes(root(res2d_wrapper.tree)) == value);
        auto res3d_wrapper = component_tree_tree_of_shapes_image3d(image3d, tos_padding::none, true, immersion,
                                                                   exterior3d);
        REQUIRE(res3d_wrapper.altitudes(root(res3d_wrapper.tree)) == value);
    }
}

TEST_CASE("test tree of shapes mean padding thin volumes", "[tree_of_shapes]") {
    const std::vector<array_shape> shapes{{4, 1, 1}, {4, 2, 1}, {4, 1, 2}, {4, 2, 2}, {4, 3, 1}, {4, 1, 3}};
    for (const auto &shape: shapes) {
        array_3d<float> image = array_3d<float>::from_shape(shape);
        std::fill(image.begin(), image.end(), 8.f);
        CAPTURE(shape[0], shape[1], shape[2]);
        REQUIRE(tree_of_shapes_internal::get_padding_value<float>(image, tos_padding::mean) == 8.f);
    }

    // every voxel of a volume of width one is on its boundary: the padding value is the mean of the volume
    array_3d<float> image = array_3d<float>::from_shape({4, 2, 1});
    for (index_t i = 0; i < (index_t) image.size(); i++) {
        image.data()[i] = (float) i;
    }
    REQUIRE(tree_of_shapes_internal::get_padding_value<float>(image, tos_padding::mean) == 3.5f);
}

TEMPLATE_TEST_CASE("test tree of shapes mean padding integer extremes", "[tree_of_shapes]",
                   std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t,
                   std::int64_t, std::uint64_t) {
    using tree_of_shapes_internal::get_padding_value;
    for (TestType value: {std::numeric_limits<TestType>::min(), std::numeric_limits<TestType>::max(), (TestType) 1}) {
        CAPTURE((long double) value);
        array_2d<TestType> image2d = array_2d<TestType>::from_shape({3, 4});
        std::fill(image2d.begin(), image2d.end(), value);
        array_3d<TestType> image3d = array_3d<TestType>::from_shape({3, 3, 4});
        std::fill(image3d.begin(), image3d.end(), value);
        if (sizeof(TestType) < 8) {
            REQUIRE(get_padding_value<TestType>(image2d, tos_padding::mean) == value);
            REQUIRE(get_padding_value<TestType>(image3d, tos_padding::mean) == value);
        } else if (value == (TestType) 1) {
            // 64-bit integers are summed in their own type: only values that do not overflow are exact
            REQUIRE(get_padding_value<TestType>(image2d, tos_padding::mean) == value);
            REQUIRE(get_padding_value<TestType>(image3d, tos_padding::mean) == value);
        }
    }
}

TEMPLATE_TEST_CASE("test tree of shapes mean padding signed rounding", "[tree_of_shapes]",
                   std::int8_t, std::int16_t, std::int32_t, std::int64_t, float, double) {
    using tree_of_shapes_internal::get_padding_value;
    // boundary sum -15 over 4 pixels
    array_2d<TestType> image{{-4, -4},
                             {-4, -3}};
    const TestType expected = (std::is_integral<TestType>::value) ? (TestType) -3 : (TestType) -3.75;
    REQUIRE(get_padding_value<TestType>(image, tos_padding::mean) == expected);
    array_2d<TestType> positive{{4, 4},
                                {4, 3}};
    REQUIRE(get_padding_value<TestType>(positive, tos_padding::mean) == -expected);

    array_3d<TestType> volume = array_3d<TestType>::from_shape({3, 3, 3});
    std::fill(volume.begin(), volume.end(), (TestType) -7);
    REQUIRE(get_padding_value<TestType>(volume, tos_padding::mean) == (TestType) -7);
}

}