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
#include <type_traits>
#include "higra/accumulator/tree_contour_accumulator.hpp"
#include "higra/image/graph_image.hpp"
#include "higra/attribute/tree_attribute.hpp"

namespace test_tree_contour_accumulator {

    using namespace hg;
    using namespace std;


    TEST_CASE("contour accumulator partition tree", "[tree_contour_accumulator]") {

        auto graph = get_4_adjacency_graph({3, 3});
        array_1d<index_t> parents{9, 9, 10, 11, 11, 13, 12, 12, 13, 10, 14, 14, 15, 14, 15, 15};
        auto tree = hg::tree(parents);

        auto depth = attribute_depth(tree);

        array_1d<double> node_saliency{0, 0, 0, 0, 0, 0, 20, 0, 0,
                                       5, 2, 7, 3, 8, 1, 50};

        auto result = accumulate_on_contours(graph, tree, node_saliency, depth, hg::accumulator_max());
        array_1d<double> expected{0, 7, 5, 7, 8, 0, 20, 8, 7, 0, 20, 8};
        REQUIRE(hg::allclose(result, expected));

        array_2d<double> node_saliency2{{0,  0},
                                        {1,  0},
                                        {2,  0},
                                        {3,  0},
                                        {4,  0},
                                        {5,  0},
                                        {6,  20},
                                        {7,  0},
                                        {8,  0},
                                        {9,  5},
                                        {10, 2},
                                        {11, 7},
                                        {12, 3},
                                        {13, 8},
                                        {14, 1},
                                        {15, 50}};

        auto result2 = accumulate_on_contours(graph, tree, node_saliency2, depth, hg::accumulator_sum());
        array_2d<double> expected2{{1,  0},
                                   {33,  14},
                                   {12,  5},
                                   {35,  14},
                                   {30,  10},
                                   {7,  0},
                                   {46,  31},
                                   {33,  15},
                                   {48,  11},
                                   {13, 0},
                                   {13, 20},
                                   {54, 12}};

        REQUIRE(hg::allclose(result2, expected2));
    }

    TEST_CASE("contour accumulator component tree", "[tree_contour_accumulator]") {

            auto graph = get_4_adjacency_graph({3, 3});
            array_1d<index_t> parents{9, 10, 10, 11, 12, 17, 14, 16, 15, 10, 17, 13, 13, 17, 16, 16, 17, 17};
            auto tree = hg::tree(parents);

            auto depth = attribute_depth(tree);

            array_1d<double> node_saliency{0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           5, 2, 8, 1, 3, 9, 2, 8, 20};

            auto result = accumulate_on_contours(graph, tree, node_saliency, depth, hg::accumulator_max());
            array_1d<double> expected{5, 8, 0, 3, 2, 8, 9, 3, 8, 8, 9, 2};
            REQUIRE(hg::allclose(result, expected));

    }

    TEST_CASE("contour accumulator index results", "[tree_contour_accumulator]") {
        auto graph = get_4_adjacency_graph({3, 3});
        array_1d<index_t> parents{9, 9, 10, 11, 11, 13, 12, 12, 13, 10, 14, 14, 15, 14, 15, 15};
        auto tree = hg::tree(parents);
        auto depth = attribute_depth(tree);
        array_2d<double> node_weights = hg::zeros<double>({16, 2});
        for (index_t i = 0; i < 16; i++) {
            node_weights(i, 0) = (double) -i;
            node_weights(i, 1) = (double) -i;
        }

        auto counter = accumulate_on_contours(graph, tree, node_weights, depth, hg::accumulator_counter());
        auto argmin = accumulate_on_contours(graph, tree, node_weights, depth, hg::accumulator_argmin());
        static_assert(std::is_same<typename decltype(counter)::value_type, index_t>::value);
        static_assert(std::is_same<typename decltype(argmin)::value_type, index_t>::value);
        REQUIRE((counter.dimension() == 1 && counter.shape()[0] == 12));
        REQUIRE((argmin.dimension() == 1 && argmin.shape()[0] == 12));
        // the two leaves of edge 0 (vertices 0 and 1) are siblings: the path contains both leaves only
        REQUIRE(counter(0) == 2);
        REQUIRE(argmin(0) == 1);
        for (index_t i = 0; i < 12; i++) {
            REQUIRE(counter(i) >= 2);
            REQUIRE(argmin(i) >= 0);
            REQUIRE(argmin(i) < counter(i));
        }
    }
}
