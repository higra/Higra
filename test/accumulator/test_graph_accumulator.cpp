/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <boost/test/unit_test.hpp>
#include "../test_utils.hpp"
#include "higra/accumulator/graph_accumulator.hpp"
#include "higra/image/graph_image.hpp"


BOOST_AUTO_TEST_SUITE(test_graph_accumulator);

    using namespace hg;
    using namespace std;


    BOOST_AUTO_TEST_CASE(test_accumulate_graph_vertices) {

        ugraph g = get_4_adjacency_graph({2, 3});

        array_1d<int> vertex_weights{1, 2, 3, 4, 5, 6};
        auto res1 = accumulate_graph_vertices(g, vertex_weights, accumulator_max());
        array_1d<int> ref1{4, 5, 6, 5, 6, 5};
        BOOST_CHECK(xt::allclose(ref1, res1));

        array_2d<int> vertex_weights2{{1, 6},
                                      {2, 5},
                                      {3, 4},
                                      {4, 3},
                                      {5, 2},
                                      {6, 1}};
        auto res2 = accumulate_graph_vertices(g, vertex_weights2, accumulator_sum());
        array_2d<int> ref2{{6,  8},
                           {9,  12},
                           {8,  6},
                           {6,  8},
                           {12, 9},
                           {8,  6}};
        BOOST_CHECK(xt::allclose(ref2, res2));
    }

    BOOST_AUTO_TEST_CASE(test_accumulate_graph_edges) {

        ugraph g = get_4_adjacency_graph({2, 3});

        array_1d<int> edge_weights{1, 2, 3, 4, 6, 5, 7};
        auto res1 = accumulate_graph_edges(g, edge_weights, accumulator_max());
        array_1d<int> ref1{2, 4, 6, 5, 7, 7};
        BOOST_CHECK(xt::allclose(ref1, res1));

        array_2d<int> edge_weights2{{1, 6},
                                      {2, 5},
                                      {3, 4},
                                      {4, 3},
                                      {5, 2},
                                      {6, 1},
                                      {7, 9}};
        auto res2 = accumulate_graph_edges(g, edge_weights2, accumulator_sum());
        array_2d<int> ref2{{3,  11},
                           {8,  13},
                           {8,  6},
                           {8,  6},
                           {17, 13},
                           {12, 11}};
        BOOST_CHECK(xt::allclose(ref2, res2));
    }


BOOST_AUTO_TEST_SUITE_END();