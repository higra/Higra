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
import numpy as np
import higra as hg


class TestTreeContourAccumulator(unittest.TestCase):

    def test_contour_accumulator_partition_tree(self):
        graph = hg.get_4_adjacency_graph((3, 3))
        tree = hg.Tree((9, 9, 10, 11, 11, 13, 12, 12, 13, 10, 14, 14, 15, 14, 15, 15))
        hg.CptHierarchy.link(tree, graph)

        node_saliency = np.asarray((0, 0, 0, 0, 0, 0, 20, 0, 0,
                                    5, 2, 7, 3, 8, 1, 50))

        result = hg.accumulate_on_contours(tree, node_saliency, hg.Accumulators.max)
        expected = np.asarray((0, 7, 5, 7, 8, 0, 20, 8, 7, 0, 20, 8))
        self.assertTrue(np.allclose(result, expected))

        node_saliency = np.asarray(((0, 0),
                                    (1, 0),
                                    (2, 0),
                                    (3, 0),
                                    (4, 0),
                                    (5, 0),
                                    (6, 20),
                                    (7, 0),
                                    (8, 0),
                                    (9, 5),
                                    (10, 2),
                                    (11, 7),
                                    (12, 3),
                                    (13, 8),
                                    (14, 1),
                                    (15, 50)))

        result = hg.accumulate_on_contours(tree, node_saliency, hg.Accumulators.sum)
        expected = np.asarray(((1, 0),
                               (33, 14),
                               (12, 5),
                               (35, 14),
                               (30, 10),
                               (7, 0),
                               (46, 31),
                               (33, 15),
                               (48, 11),
                               (13, 0),
                               (13, 20),
                               (54, 12)))
        self.assertTrue(np.allclose(result, expected))

    def test_contour_accumulator_component_tree(self):
        graph = hg.get_4_adjacency_graph((3, 3))
        tree = hg.Tree((9, 10, 10, 11, 12, 17, 14, 16, 15, 10, 17, 13, 13, 17, 16, 16, 17, 17))
        hg.CptHierarchy.link(tree, graph)

        node_saliency = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    5, 2, 8, 1, 3, 9, 2, 8, 20))

        result = hg.accumulate_on_contours(tree, node_saliency, hg.Accumulators.max)
        expected = np.asarray((5, 8, 0, 3, 2, 8, 9, 3, 8, 8, 9, 2))
        self.assertTrue(np.allclose(result, expected))


if __name__ == '__main__':
    unittest.main()
