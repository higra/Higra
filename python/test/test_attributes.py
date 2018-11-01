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


class TestAttributes(unittest.TestCase):

    @staticmethod
    def get_test_tree():
        """
        base graph is

        (0)-- 0 --(1)-- 2 --(2)
         |         |         |
         6         6         0
         |         |         |
        (3)-- 0 --(4)-- 4 --(5)
         |         |         |
         5         5         3
         |         |         |
        (6)-- 0 --(7)-- 1 --(8)

        Minima are
        A: (0,1)
        B: (3,4)
        C: (2,5)
        D: (6,7)

        BPT:




        4                 +-------16------+
                          |               |
        3         +-------15-----+        |
                  |              |        |
        2     +---14--+          |        |
              |       |          |        |
        1     |       |       +--13-+     |
              |       |       |     |     |
        0   +-9-+   +-10+   +-12+   |   +-11+
            +   +   +   +   +   +   +   +   +
            0   1   2   5   6   7   8   3   4


        :return:
        """

        g = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0, 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1))
        hg.CptEdgeWeightedGraph.link(edge_weights, g)

        return hg.bpt_canonical(edge_weights)

    def setUp(self):
        hg.clear_all_attributes()

    def test_area(self):
        tree, altitudes = TestAttributes.get_test_tree()

        ref_area = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 7, 9]
        area = hg.attribute_area(tree)
        self.assertTrue(np.allclose(ref_area, area))

        leaf_area = np.asarray([1, 2, 1, 1, 2, 1, 1, 1, 3])
        ref_area = [1, 2, 1, 1, 2, 1, 1, 1, 3, 3, 2, 3, 2, 5, 5, 10, 13]
        area = hg.attribute_area(tree, leaf_area=leaf_area, force_recompute=True)
        self.assertTrue(np.allclose(ref_area, area))

    def test_volume(self):
        tree, altitudes = TestAttributes.get_test_tree()

        ref_attribute = [0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 8, 2, 9, 12, 28, 36]
        attribute = hg.attribute_volume(altitudes)
        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_lca_map(self):
        tree, altitudes = TestAttributes.get_test_tree()

        ref_attribute = [9, 16, 14, 16, 10, 11, 16, 16, 16, 15, 12, 13]
        attribute = hg.attribute_lca_map(tree)
        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_frontier_length(self):
        tree, altitudes = TestAttributes.get_test_tree()

        ref_attribute = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 5]
        attribute = hg.attribute_frontier_length(tree)
        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_perimeter_length(self):
        tree, altitudes = TestAttributes.get_test_tree()

        ref_attribute = [2, 3, 2, 3, 4, 3, 2, 3, 2, 3, 3, 5, 3, 3, 4, 5, 0]
        attribute = hg.attribute_perimeter_length(tree)
        self.assertTrue(np.allclose(ref_attribute, attribute))

        leaf_perimeter = 4 + np.zeros((tree.num_leaves(),))
        ref_attribute = [4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 8, 10, 16, 12]
        attribute = hg.attribute_perimeter_length(tree, leaf_perimeter=leaf_perimeter, force_recompute=True)
        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_compactness(self):
        tree, altitudes = TestAttributes.get_test_tree()

        hg.set_attribute(hg.get_attribute(tree, "leaf_graph"), "vertex_perimeter",
                         4 + np.zeros((tree.num_leaves(),), dtype=np.int64))
        ref_attribute = [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.88888889, 0.88888889, 0.88888889, 0.88888889, 0.75, 0.64,
                         0.4375, 1.]
        attribute = hg.attribute_compactness(tree)
        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_mean_weights(self):
        tree, altitudes = TestAttributes.get_test_tree()

        leaf_data = np.asarray(((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)),
                               dtype=np.float64)
        ref_attribute = np.asarray(((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
                                    (1. / 2, 1. / 2), (7. / 2, 7. / 2), (7. / 2, 7. / 2), (13. / 2, 13. / 2), (7., 7.),
                                    (2., 2.), (29. / 7, 29. / 7), (4., 4.)))

        attribute = hg.attribute_mean_weights(tree, leaf_data=leaf_data)
        self.assertTrue(np.allclose(ref_attribute, attribute))


if __name__ == '__main__':
    unittest.main()
