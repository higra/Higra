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
#include "test_utils.hpp"
#include "higra/algo/graph_core.hpp"
#include "higra/utils.hpp"
#include "higra/image/graph_image.hpp"

using namespace hg;

BOOST_AUTO_TEST_SUITE(test_cut);


    BOOST_AUTO_TEST_CASE(test_graph_cut_2_labelisation) {
        auto graph = get_4_adjacency_graph({3, 3});
        array_1d<char> edge_weights={1,0,0,1,1,0,0,1,1,0,1,0};

        auto labels = graph_cut_2_labelisation(graph, edge_weights);

        array_1d<index_t> ref_labels = {1, 2, 2, 1, 1, 3, 1, 3, 3};
        BOOST_CHECK(is_in_bijection(labels, ref_labels));
    }

    BOOST_AUTO_TEST_CASE(test_labelisation_2_graph_cut) {
            auto graph = get_4_adjacency_graph({3, 3});
            array_1d<index_t> labels = {1, 2, 2, 1, 1, 3, 1, 3, 3};

            auto edge_weights = labelisation_2_graph_cut(graph, labels);

            array_1d<char> ref_edge_weights={1,0,0,1,1,0,0,1,1,0,1,0};

            BOOST_CHECK(is_in_bijection(edge_weights, ref_edge_weights));
    }

BOOST_AUTO_TEST_SUITE_END();


