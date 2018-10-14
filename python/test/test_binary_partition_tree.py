############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import unittest
import higra as hg
import numpy as np


class TestBinaryPartitionTree(unittest.TestCase):

    def test_binary_partition_tree_complete_linkage(self):
        graph = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((1, 8, 2, 10, 15, 3, 11, 4, 12, 13, 5, 6), np.float32)
        res = hg._binary_partition_tree_complete_linkage(graph, edge_weights)
        tree = res.tree()
        levels = res.node_altitude()

        expected_parents = np.asarray((9, 9, 10, 11, 11, 12, 13, 13, 14, 10, 16, 12, 15, 14, 15, 16, 16), np.uint32)
        expected_levels = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 13, 15), np.float32)

        self.assertTrue(np.all(expected_parents == tree.parents()))
        self.assertTrue(np.all(expected_levels == levels))

    def test_binary_partition_tree_average_linkage(self):
        graph = hg.get_4_adjacency_graph((3, 3))
        edge_values = np.asarray((1, 7, 2, 10, 16, 3, 11, 4, 12, 14, 5, 6), np.float32)
        edge_weights = np.asarray((7, 1, 7, 3, 2, 8, 2, 2, 2, 1, 5, 9), np.float32)
        res = hg._binary_partition_tree_average_linkage(graph, edge_values, edge_weights)
        tree = res.tree()
        levels = res.node_altitude()

        expected_parents = np.asarray((9, 9, 10, 11, 11, 12, 13, 13, 14, 10, 15, 12, 15, 14, 16, 16, 16), np.uint32)
        expected_levels = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 11.5, 12), np.float32)

        self.assertTrue(np.all(expected_parents == tree.parents()))
        self.assertTrue(np.all(expected_levels == levels))

    def test_binary_partition_tree_custom_linkage(self):
        graph = hg.get_4_adjacency_graph((3, 3))
        edge_values = np.asarray((1, 7, 2, 10, 16, 3, 11, 4, 12, 14, 5, 6), np.float32)
        edge_weights = np.asarray((7, 1, 7, 3, 2, 8, 2, 2, 2, 1, 5, 9), np.float32)

        def weighting_function_average_linkage(graph, fusion_edge_index, new_region, merged_region1, merged_region2, new_neighbours):
            for n in new_neighbours:
                if n.num_edges() > 1:
                    new_weight = edge_weights[n.first_edge_index()] + edge_weights[n.second_edge_index()]
                    new_value = (edge_values[n.first_edge_index()] * edge_weights[n.first_edge_index()] \
                        + edge_values[n.second_edge_index()] * edge_weights[n.second_edge_index()]) \
                        / new_weight
                else:
                    new_weight = edge_weights[n.first_edge_index()]
                    new_value = edge_values[n.first_edge_index()]

                n.set_new_edge_weight(new_value)
                edge_values[n.new_edge_index()] = new_value
                edge_weights[n.new_edge_index()] = new_weight

        res = hg._binary_partition_tree(graph, edge_values, weighting_function_average_linkage)
        tree = res.tree()
        levels = res.node_altitude()

        expected_parents = np.asarray((9, 9, 10, 11, 11, 12, 13, 13, 14, 10, 15, 12, 15, 14, 16, 16, 16), np.uint32)
        expected_levels = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 11.5, 12), np.float32)

        self.assertTrue(np.all(expected_parents == tree.parents()))
        self.assertTrue(np.all(expected_levels == levels))


if __name__ == '__main__':
    unittest.main()
