//
// Created by perretb on 16/05/18.
//

#include <boost/test/unit_test.hpp>
#include "test_utils.hpp"
#include "higra/algo/watershed.hpp"
#include "higra/algo/graph_image.hpp"

using namespace hg;

BOOST_AUTO_TEST_SUITE(algo_watershed);


    BOOST_AUTO_TEST_CASE(watershed_simple) {

        // Fig 4 of Watershed Cuts: Minimum Spanning Forests and the
        // Drop of Water Principle
        // Jean Cousty, Gilles Bertrand, Laurent Najman, Michel Couprie
        auto g = hg::get_4_adjacency_graph({4, 4});
        array_1d<int> edge_weights{1, 2, 5, 5, 5, 8, 1, 4, 3, 4, 4, 1, 5, 2, 6, 3, 5, 4, 0, 7, 0, 3, 4, 0};

        auto labels = hg::labelisation_watershed(g, edge_weights);

        array_1d<int> expected{1, 1, 1, 2,
                               1, 1, 2, 2,
                               1, 1, 3, 3,
                               1, 1, 3, 3};
        BOOST_CHECK(labels == expected);
    }


BOOST_AUTO_TEST_SUITE_END();