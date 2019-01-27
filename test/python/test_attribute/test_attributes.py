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

    def test_area(self):
        g = hg.get_4_adjacency_graph((2, 3))
        edge_weights = np.asarray((1, 4, 6, 5, 2, 7, 3))

        ref_area = (1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 6)

        tree, altitudes = hg.bpt_canonical(edge_weights, g)
        area = hg.attribute_area(tree)
        self.assertTrue(np.all(ref_area == area))

        tree2 = hg.Tree(tree.parents())
        area2 = hg.attribute_area(tree2)
        self.assertTrue(np.all(ref_area == area2))

    def test_sibling(self):
        t = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7))
        ref = np.asarray((1, 0, 3, 4, 2, 6, 5, 7))
        res = hg.attribute_sibling(t)
        self.assertTrue(np.all(ref == res))

        t = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7))
        ref2 = np.asarray((1, 0, 4, 2, 3, 6, 5, 7))
        res2 = hg.attribute_sibling(t, -1)
        self.assertTrue(np.all(ref2 == res2))

    def test_depth(self):
        t = hg.Tree((6, 6, 7, 8, 8, 8, 7, 9, 9, 9))
        ref = np.asarray((3, 3, 2, 2, 2, 2, 2, 1, 1, 0))
        res = hg.attribute_depth(t)
        self.assertTrue(np.all(ref == res))


if __name__ == '__main__':
    unittest.main()
