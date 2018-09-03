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
        tree, levels = hg._binary_partition_tree_complete_linkage(graph, edge_weights)

        expected_parents = np.asarray((9, 9, 10, 11, 11, 12, 13, 13, 14, 10, 16, 12, 15, 14, 15, 16, 16), np.uint32)
        expected_levels = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 13, 15), np.float32)

        self.assertTrue(np.all(expected_parents == tree.parents()))
        self.assertTrue(np.all(expected_levels == levels))

    def test_binary_partition_tree_average_linkage(self):
        graph = hg.get_4_adjacency_graph((3, 3))
        edge_values = np.asarray((1, 7, 2, 10, 16, 3, 11, 4, 12, 14, 5, 6), np.float32)
        edge_weights = np.asarray((7, 1, 7, 3, 2, 8, 2, 2, 2, 1, 5, 9), np.float32)
        tree, levels = hg._binary_partition_tree_average_linkage(graph, edge_values, edge_weights)

        expected_parents = np.asarray((9, 9, 10, 11, 11, 12, 13, 13, 14, 10, 15, 12, 15, 14, 16, 16, 16), np.uint32)
        expected_levels = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 11.5, 12), np.float32)

        self.assertTrue(np.all(expected_parents == tree.parents()))
        self.assertTrue(np.all(expected_levels == levels))


if __name__ == '__main__':
    unittest.main()
