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
#include "higra/hierarchy/binary_partition_tree.hpp"
#include "higra/image/graph_image.hpp"
#include "xtensor/xrandom.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "higra/algo/tree.hpp"


using namespace hg;
using namespace std;

BOOST_AUTO_TEST_SUITE(test_binary_partition_tree);

    BOOST_AUTO_TEST_CASE(test_single_linkage_clustering_simple) {
        auto graph = get_4_adjacency_graph({3, 3});
        array_1d<double> edge_weights({1, 9, 6, 7, 5, 8, 12, 4, 10, 11, 2, 3});
        auto res = hg::binary_partition_tree(graph,
                                             edge_weights,
                                             hg::make_binary_partition_tree_min_linkage(
                                                     edge_weights));
        auto &tree = res.tree;
        auto &levels = res.altitudes;

        array_1d<index_t> expected_parents({9, 9, 13, 15, 12, 12, 10, 10, 11, 14, 11, 16, 13, 14, 15, 16, 16});
        array_1d<double> expected_levels({0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 8, 10});

        BOOST_CHECK(expected_parents == tree.parents());
        BOOST_CHECK(expected_levels == levels);
    }

    BOOST_AUTO_TEST_CASE(test_single_linkage_clustering_hard) {
        long size = 100;
        auto graph = get_4_adjacency_graph({size, size});
        array_1d<double> edge_weights = xt::random::rand<double>({num_edges(graph)});
        auto res = hg::binary_partition_tree(graph,
                                             edge_weights,
                                             hg::make_binary_partition_tree_min_linkage(edge_weights));
        auto &tree = res.tree;


        auto res2 = hg::bpt_canonical(graph, edge_weights);
        auto &tree2 = res2.tree;

        BOOST_CHECK(hg::test_tree_isomorphism(tree, tree2));
    }

    BOOST_AUTO_TEST_CASE(test_complete_linkage_clustering_simple) {
        auto graph = get_4_adjacency_graph({3, 3});
        array_1d<double> edge_weights({1, 8, 2, 10, 15, 3, 11, 4, 12, 13, 5, 6});
        auto res = hg::binary_partition_tree(graph,
                                             edge_weights,
                                             hg::make_binary_partition_tree_complete_linkage(
                                                     edge_weights));
        auto &tree = res.tree;
        auto &levels = res.altitudes;

        array_1d<index_t> expected_parents({9, 9, 10, 11, 11, 12, 13, 13, 14, 10, 16, 12, 15, 14, 15, 16, 16});
        array_1d<double> expected_levels({0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 13, 15});

        BOOST_CHECK(expected_parents == tree.parents());
        BOOST_CHECK(expected_levels == levels);
    }

    BOOST_AUTO_TEST_CASE(test_average_linkage_clustering_simple) {
        auto graph = get_4_adjacency_graph({3, 3});
        array_1d<double> edge_values({1, 7, 2, 10, 16, 3, 11, 4, 12, 14, 5, 6});
        array_1d<double> edge_weights({7, 1, 7, 3, 2, 8, 2, 2, 2, 1, 5, 9});
        auto res = hg::binary_partition_tree(graph,
                                             edge_values,
                                             hg::make_binary_partition_tree_average_linkage(
                                                     edge_values, edge_weights));
        auto &tree = res.tree;
        auto &levels = res.altitudes;

        array_1d<index_t> expected_parents({9, 9, 10, 11, 11, 12, 13, 13, 14, 10, 15, 12, 15, 14, 16, 16, 16});
        array_1d<double> expected_levels({0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 11.5, 12});

        BOOST_CHECK(expected_parents == tree.parents());
        BOOST_CHECK(expected_levels == levels);
    }


BOOST_AUTO_TEST_SUITE_END();
