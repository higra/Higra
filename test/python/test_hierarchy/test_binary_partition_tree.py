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
        tree, levels = hg.binary_partition_tree_complete_linkage(graph, edge_weights)

        expected_parents = np.asarray((9, 9, 10, 11, 11, 12, 13, 13, 14, 10, 16, 12, 15, 14, 15, 16, 16), np.uint32)
        expected_levels = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 13, 15), np.float32)

        self.assertTrue(np.all(expected_parents == tree.parents()))
        self.assertTrue(np.all(expected_levels == levels))

    def test_binary_partition_tree_average_linkage(self):
        graph = hg.get_4_adjacency_graph((3, 3))
        edge_values = np.asarray((1, 7, 2, 10, 16, 3, 11, 4, 12, 14, 5, 6), np.float32)
        edge_weights = np.asarray((7, 1, 7, 3, 2, 8, 2, 2, 2, 1, 5, 9), np.float32)
        tree, levels = hg.binary_partition_tree_average_linkage(graph, edge_values, edge_weights)

        expected_parents = np.asarray((9, 9, 10, 11, 11, 12, 13, 13, 14, 10, 15, 12, 15, 14, 16, 16, 16), np.uint32)
        expected_levels = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 11.5, 12), np.float32)

        self.assertTrue(np.all(expected_parents == tree.parents()))
        self.assertTrue(np.all(expected_levels == levels))

    def test_binary_partition_tree_custom_linkage(self):
        graph = hg.get_4_adjacency_graph((3, 3))
        edge_values = np.asarray((1, 7, 2, 10, 16, 3, 11, 4, 12, 14, 5, 6), np.float32)
        edge_weights = np.asarray((7, 1, 7, 3, 2, 8, 2, 2, 2, 1, 5, 9), np.float32)

        def weighting_function_average_linkage(graph, fusion_edge_index, new_region, merged_region1, merged_region2,
                                               new_neighbours):
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

        tree, altitudes = hg.binary_partition_tree(graph, weighting_function_average_linkage, edge_values)

        expected_parents = np.asarray((9, 9, 10, 11, 11, 12, 13, 13, 14, 10, 15, 12, 15, 14, 16, 16, 16), np.uint32)
        expected_altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 11.5, 12), np.float32)

        self.assertTrue(np.all(expected_parents == tree.parents()))
        self.assertTrue(np.all(expected_altitudes == altitudes))

    def test_binary_partition_tree_average_linkage2(self):
        graph = hg.UndirectedGraph(10)
        graph.add_edges((0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 7, 7),
                        (3, 6, 4, 2, 5, 3, 6, 9, 7, 3, 8, 5, 9, 4, 6, 9, 7, 8, 6, 9, 8))
        edge_values = np.asarray((0.87580029, 0.60123697, 0.79924759, 0.74221387, 0.75418382, 0.66159356,
                                  1.31856839, 0.76080612, 1.08881471, 0.98557615, 0.61454158, 0.50913424,
                                  0.63556478, 0.64684775, 1.14865302, 0.81741018, 2.1591071, 0.60563004,
                                  2.06636665, 1.35617725, 0.83085949), dtype=np.float64)

        tree, altitudes = hg.binary_partition_tree_average_linkage(graph, edge_values)

        expected_parents = np.asarray((11, 14, 10, 13, 15, 10, 11, 18, 12, 13, 12, 17, 16, 14, 15, 16, 17, 18, 18),
                                      dtype=np.int64)
        expected_altitudes = np.asarray((0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.509134, 0.601237, 0.610086,
                                         0.635565, 0.661594, 0.732129, 0.810695, 1.241727, 1.35874), dtype=np.float64)
        self.assertTrue(np.all(expected_parents == tree.parents()))
        self.assertTrue(np.allclose(expected_altitudes, altitudes))

    def test_binary_partition_tree_ward_linkage(self):
        graph = hg.UndirectedGraph(5)

        graph.add_edges((0, 0, 0, 1, 2, 2, 3), (1, 2, 3, 2, 3, 4, 4))

        vertex_centroids = np.asarray(((0, 0),
                                       (1, 1),
                                       (1, 3),
                                       (-3, 4),
                                       (-1, 5)))

        vertex_sizes = np.asarray((1, 1, 1, 2, 1))

        tree, altitudes = hg.binary_partition_tree_ward_linkage(graph, vertex_centroids, vertex_sizes)

        expected_parents = np.asarray((5, 5, 7, 6, 6, 7, 8, 8, 8), dtype=np.int64)
        expected_altitudes = np.asarray((0., 0., 0., 0., 0.,
                                         1., 3.333333, 4.333333, 27.), dtype=np.float64)
        self.assertTrue(np.all(expected_parents == tree.parents()))
        self.assertTrue(np.allclose(expected_altitudes, altitudes))


if __name__ == '__main__':
    unittest.main()
