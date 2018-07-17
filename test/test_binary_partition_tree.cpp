#include <boost/test/unit_test.hpp>
#include "test_utils.hpp"
#include "higra/hierarchy/binary_partition_tree.hpp"
#include "higra/algo/graph_image.hpp"
#include "xtensor/xrandom.hpp"
//
// Created by user on 3/9/18.
//


using namespace hg;
using namespace std;

BOOST_AUTO_TEST_SUITE(test_binary_partition_tree);

    BOOST_AUTO_TEST_CASE(test_single_linkage_clustering) {
        auto graph = get_4_adjacency_graph({3, 3});
        array_1d<double> edge_weights = xt::random::rand<double>({num_edges(graph)});
        hg::binary_partition_tree_internal::binary_partition_tree(graph, edge_weights);
    }


BOOST_AUTO_TEST_SUITE_END();
