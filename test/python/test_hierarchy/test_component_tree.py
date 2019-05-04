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

    def test_component_tree_min_tree(self):
        graph = hg.get_4_adjacency_implicit_graph((4, 4))
        vertex_weights = -1. * np.asarray(((0, 1, 4, 4),
                                           (7, 5, 6, 8),
                                           (2, 3, 4, 1),
                                           (9, 8, 6, 7)), dtype=np.float64)
        tree, altitudes = hg.component_tree_min_tree(vertex_weights, graph)

        expected_parents = np.asarray((28, 27, 24, 24,
                                       20, 23, 22, 18,
                                       26, 25, 24, 27,
                                       16, 17, 21, 19,
                                       17, 21, 22, 21, 23, 24, 23, 24, 25, 26, 27, 28, 28), np.int64)
        expected_altitudes = -1. * np.asarray((0., 1., 4., 4.,
                                               7., 5., 6., 8.,
                                               2., 3., 4., 1.,
                                               9., 8., 6., 7., 9.,
                                               8., 8., 7., 7., 6.,
                                               6., 5., 4., 3., 2.,
                                               1., 0.), np.float64)

        self.assertTrue(np.all(expected_parents == tree.parents()))
        self.assertTrue(np.allclose(expected_altitudes, altitudes))

    def test_component_tree_max_tree(self):
        graph = hg.get_4_adjacency_implicit_graph((4, 4))
        vertex_weights = np.asarray(((0, 1, 4, 4),
                                     (7, 5, 6, 8),
                                     (2, 3, 4, 1),
                                     (9, 8, 6, 7)), dtype=np.float64)
        tree, altitudes = hg.component_tree_max_tree(vertex_weights, graph)

        expected_parents = np.asarray((28, 27, 24, 24,
                                       20, 23, 22, 18,
                                       26, 25, 24, 27,
                                       16, 17, 21, 19,
                                       17, 21, 22, 21, 23, 24, 23, 24, 25, 26, 27, 28, 28), np.int64)
        expected_altitudes = np.asarray((0., 1., 4., 4.,
                                         7., 5., 6., 8.,
                                         2., 3., 4., 1.,
                                         9., 8., 6., 7., 9.,
                                         8., 8., 7., 7., 6.,
                                         6., 5., 4., 3., 2.,
                                         1., 0.), np.float64)

        self.assertTrue(np.all(expected_parents == tree.parents()))
        self.assertTrue(np.allclose(expected_altitudes, altitudes))


if __name__ == '__main__':
    unittest.main()
