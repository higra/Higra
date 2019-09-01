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
#include "higra/accumulator/tree_contour_accumulator.hpp"
#include "higra/image/graph_image.hpp"
#include "higra/attribute/tree_attribute.hpp"


using namespace hg;
using namespace std;

namespace tree_contour_accumulator {


    TEST_CASE("contour accumulator partition tree", "[tree_contour_accumulator]") {

        auto graph = get_4_adjacency_graph({3, 3});
        array_1d<index_t> parents{9, 9, 10, 11, 11, 13, 12, 12, 13, 10, 14, 14, 15, 14, 15, 15};
        auto tree = hg::tree(parents);

        auto depth = attribute_depth(tree);

        array_1d<double> node_saliency{0, 0, 0, 0, 0, 0, 20, 0, 0,
                                       5, 2, 7, 3, 8, 1, 50};

        auto result = accumulate_on_contours(graph, tree, node_saliency, depth, hg::accumulator_max());
        array_1d<double> expected{0, 7, 5, 7, 8, 0, 20, 8, 7, 0, 20, 8};
        REQUIRE(xt::allclose(result, expected));

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

        REQUIRE(xt::allclose(result2, expected2));
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
            REQUIRE(xt::allclose(result, expected));

    }
}