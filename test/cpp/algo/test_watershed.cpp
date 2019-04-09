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
#include "higra/algo/watershed.hpp"
#include "higra/image/graph_image.hpp"

using namespace hg;

namespace test_watershed {

    TEST_CASE("watershed cut simple", "[watershed_cut]") {

        // Fig 4 of Watershed Cuts: Minimum Spanning Forests and the
        // Drop of Water Principle
        // Jean Cousty, Gilles Bertrand, Laurent Najman, Michel Couprie
        auto g = hg::get_4_adjacency_graph({4, 4});
        array_1d<int> edge_weights{1, 2, 5, 5, 5, 8, 1, 4, 3, 4, 4, 1, 5, 2, 6, 3, 5, 4, 0, 7, 0, 3, 4, 0};

        auto labels = hg::labelisation_watershed(g, edge_weights);

        array_1d<index_t> expected{1, 1, 1, 2,
                                   1, 1, 2, 2,
                                   1, 1, 3, 3,
                                   1, 1, 3, 3};
        REQUIRE((labels == expected));
    }

    TEST_CASE("watershed cut simple 2", "[watershed_cut]") {
        auto g = hg::get_4_adjacency_graph({3, 3});
        array_1d<int> edge_weights{1, 1, 0, 0, 0, 1, 0, 0, 2, 2, 0, 2};

        auto labels = hg::labelisation_watershed(g, edge_weights);

        array_1d<index_t> expected{1, 1, 1,
                                   2, 1, 1,
                                   2, 2, 1};
        REQUIRE((labels == expected));
    }

}
