/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "../test_utils.hpp"
#include "higra/hierarchy/component_tree.hpp"
#include "higra/attribute/tree_attribute.hpp"
#include "higra/image/graph_image.hpp"
#include "higra/algo/tree.hpp"
#include "xtensor/containers/xadapt.hpp"

using namespace hg;
using namespace std;

namespace binary_partition_tree {

    TEST_CASE("test pre_tree_construction", "[component_tree]") {
        auto graph = get_4_adjacency_implicit_graph({4, 4});
        array_1d<double> vertex_weights({0, 1, 4, 4,
                                         7, 5, 6, 8,
                                         2, 3, 4, 1,
                                         9, 8, 6, 7});
        array_1d<index_t> sorted_vertex_indices = xt::arange(num_vertices(graph));
        std::stable_sort(sorted_vertex_indices.begin(), sorted_vertex_indices.end(),
                         [&vertex_weights](index_t i, index_t j) { return vertex_weights[i] < vertex_weights[j]; });
        auto parents = component_tree_internal::pre_tree_construction(graph, sorted_vertex_indices);

        array_1d<index_t> expected_parents({0, 0, 9, 2,
                                            5, 10, 5, 6,
                                            11, 8, 3, 1,
                                            13, 14, 10, 14});
        REQUIRE((expected_parents == parents));
    }

    TEST_CASE("test canonize_tree", "[component_tree]") {
        auto graph = get_4_adjacency_implicit_graph({4, 4});
        array_1d<double> vertex_weights({0, 1, 4, 4,
                                         7, 5, 6, 8,
                                         2, 3, 4, 1,
                                         9, 8, 6, 7});
        array_1d<index_t> sorted_vertex_indices = xt::arange(num_vertices(graph));
        std::stable_sort(sorted_vertex_indices.begin(), sorted_vertex_indices.end(),
                         [&vertex_weights](index_t i, index_t j) { return vertex_weights[i] < vertex_weights[j]; });

        array_1d<index_t> parents({0, 0, 9, 2,
                                   5, 10, 5, 6,
                                   11, 8, 3, 1,
                                   13, 14, 10, 14});

        component_tree_internal::canonize_tree(parents, vertex_weights, sorted_vertex_indices);

        array_1d<index_t> expected_parents({0, 0, 9, 2,
                                            5, 2, 5, 6,
                                            1, 8, 2, 1,
                                            13, 14, 2, 14});
        REQUIRE((expected_parents == parents));
    }

    TEST_CASE("test expand_canonized_parent_relation", "[component_tree]") {
        auto graph = get_4_adjacency_implicit_graph({4, 4});
        array_1d<double> vertex_weights({0, 1, 4, 4,
                                         7, 5, 6, 8,
                                         2, 3, 4, 1,
                                         9, 8, 6, 7});
        array_1d<index_t> sorted_vertex_indices = xt::arange(num_vertices(graph));
        std::stable_sort(sorted_vertex_indices.begin(), sorted_vertex_indices.end(),
                         [&vertex_weights](index_t i, index_t j) { return vertex_weights[i] < vertex_weights[j]; });

        array_1d<index_t> parents({0, 0, 9, 2,
                                   5, 2, 5, 6,
                                   1, 8, 2, 1,
                                   13, 14, 2, 14});

        auto res = component_tree_internal::expand_canonized_parent_relation(parents, vertex_weights,
                                                                             sorted_vertex_indices);
        auto new_parents = xt::adapt(res.first, {res.first.size()});
        auto new_altitudes = xt::adapt(res.second, {res.second.size()});

        array_1d<index_t> expected_parents({28, 27, 24, 24,
                                            20, 23, 22, 18,
                                            26, 25, 24, 27,
                                            16, 17, 21, 19,
                                            17, 21, 22, 21, 23, 24, 23, 24, 25, 26, 27, 28, 28});
        REQUIRE((expected_parents == new_parents));

        array_1d<double> expected_altitudes({0., 1., 4., 4.,
                                             7., 5., 6., 8.,
                                             2., 3., 4., 1.,
                                             9., 8., 6., 7., 9.,
                                             8., 8., 7., 7., 6.,
                                             6., 5., 4., 3., 2.,
                                             1., 0.});
        REQUIRE((expected_altitudes == new_altitudes));
    }

    TEST_CASE("test max tree", "[component_tree]") {
        auto graph = get_4_adjacency_implicit_graph({4, 4});
        array_1d<double> vertex_weights({0, 1, 4, 4,
                                         7, 5, 6, 8,
                                         2, 3, 4, 1,
                                         9, 8, 6, 7});

        auto res = component_tree_max_tree(graph, vertex_weights);
        auto &tree = res.tree;
        auto &altitudes = res.altitudes;

        REQUIRE(category(tree) == tree_category::component_tree);
        array_1d<index_t> expected_parents({28, 27, 24, 24,
                                            20, 23, 22, 18,
                                            26, 25, 24, 27,
                                            16, 17, 21, 19,
                                            17, 21, 22, 21, 23, 24, 23, 24, 25, 26, 27, 28, 28});
        REQUIRE((expected_parents == tree.parents()));

        array_1d<double> expected_altitudes({0., 1., 4., 4.,
                                             7., 5., 6., 8.,
                                             2., 3., 4., 1.,
                                             9., 8., 6., 7., 9.,
                                             8., 8., 7., 7., 6.,
                                             6., 5., 4., 3., 2.,
                                             1., 0.});
        REQUIRE((expected_altitudes == altitudes));
    }

    TEST_CASE("test min tree", "[component_tree]") {
        auto graph = get_4_adjacency_implicit_graph({4, 4});
        array_1d<double> vertex_weights({0, 1, 4, 4,
                                         7, 5, 6, 8,
                                         2, 3, 4, 1,
                                         9, 8, 6, 7});

        vertex_weights *= -1.;

        auto res = component_tree_min_tree(graph, vertex_weights);
        auto &tree = res.tree;
        auto &altitudes = res.altitudes;

        REQUIRE(category(tree) == tree_category::component_tree);
        array_1d<index_t> expected_parents({28, 27, 24, 24,
                                            20, 23, 22, 18,
                                            26, 25, 24, 27,
                                            16, 17, 21, 19,
                                            17, 21, 22, 21, 23, 24, 23, 24, 25, 26, 27, 28, 28});
        REQUIRE((expected_parents == tree.parents()));

        array_1d<double> expected_altitudes({0., 1., 4., 4.,
                                             7., 5., 6., 8.,
                                             2., 3., 4., 1.,
                                             9., 8., 6., 7., 9.,
                                             8., 8., 7., 7., 6.,
                                             6., 5., 4., 3., 2.,
                                             1., 0.});
        expected_altitudes *= -1.;
        REQUIRE((expected_altitudes == altitudes));
    }

    TEST_CASE("test max tree area filter", "[component_tree]") {
        auto graph = get_4_adjacency_implicit_graph({5, 5});
        array_1d<double> vertex_weights({-5, 2, 2, 5, 5,
                                         -4, 2, 2, 6, 5,
                                         3, 3, 3, 3, 3,
                                         -2, -2, -2, 9, 7,
                                         -1, 0, -2, 8, 9});

        auto res = component_tree_max_tree(graph, vertex_weights);
        auto &tree = res.tree;
        auto &altitudes = res.altitudes;

        auto area = attribute_area(tree);
        auto filtered_weights = reconstruct_leaf_data(tree, altitudes, area <= 4);

        array_1d<double> expected_filtered_weights
                ({-5, 2, 2, 3, 3,
                  -4, 2, 2, 3, 3,
                  3, 3, 3, 3, 3,
                  -2, -2, -2, 3, 3,
                  -2, -2, -2, 3, 3});

        REQUIRE((expected_filtered_weights == filtered_weights));
    }
}