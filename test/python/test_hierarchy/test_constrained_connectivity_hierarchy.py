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


class TestConstrainedConnectivityHierarchy(unittest.TestCase):

    def test_alpha_omega_hierarchy(self):
        graph = hg.get_4_adjacency_graph((3, 3))
        vertex_weights = np.asarray(((1, 2, 3),
                                     (5, 6, 5),
                                     (22, 21, 20)))
        tree, levels = hg.constrained_connectivity_hierarchy_alpha_omega(graph, vertex_weights)

        expected_parents = np.asarray((11, 11, 11, 9, 9, 9, 10, 10, 10, 11, 12, 12, 12), np.uint32)
        expected_levels = np.asarray((0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 5., 15), np.float32)

        self.assertTrue(np.all(expected_parents == tree.parents()))
        self.assertTrue(np.allclose(expected_levels, levels))

    def test_strong_hierarchy(self):
        graph = hg.get_4_adjacency_graph((2, 5))
        edge_weights = np.asarray((1, 3, 2, 1, 15, 1, 1, 1, 5, 1, 2, 15, 1))
        tree, levels = hg.constrained_connectivity_hierarchy_strong_connection(graph, edge_weights)

        expected_parents = np.asarray((12, 12, 10, 11, 11, 12, 12, 10, 11, 11, 12, 13, 13, 13), np.uint32)
        expected_levels = np.asarray((0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 5., 3., 15), np.float32)

        self.assertTrue(np.all(expected_parents == tree.parents()))
        self.assertTrue(np.allclose(expected_levels, levels))


if __name__ == '__main__':
    unittest.main()
