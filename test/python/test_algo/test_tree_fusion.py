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


class TestWatershed(unittest.TestCase):

    def test_tree_fusion1(self):
        t1 = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7))
        t2 = hg.Tree((7, 7, 6, 5, 5, 6, 7, 7))

        res = hg.tree_fusion_depth_map((t1, t2))

        expected = np.asarray((2, 2, 2, 3, 3))

        diff = expected - res
        self.assertTrue(np.all(diff == diff[0]))

    def test_tree_fusion2(self):
        t1 = hg.Tree((4, 4, 6, 5, 6, 6, 6))
        t2 = hg.Tree((4, 5, 5, 5, 5, 5))

        res = hg.tree_fusion_depth_map((t1, t2))

        expected = np.asarray((3, 2, 1, 2))

        diff = expected - res
        self.assertTrue(np.all(diff == diff[0]))

    def test_tree_fusion3(self):
        im1 = np.asarray((0, 0, 0, 0, 0, 0, 0,
                          3, 3, 3, 2, 1, 1, 1,
                          3, 3, 3, 2, 1, 1, 1,
                          3, 3, 3, 2, 1, 1, 1,
                          2, 2, 2, 2, 1, 1, 1,
                          1, 1, 1, 1, 1, 0, 0), dtype=np.float32)

        im2 = np.asarray((0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0,
                          0, 2, 1, 1, 1, 2, 0,
                          0, 1, 1, 1, 1, 2, 0,
                          0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0), dtype=np.float32)

        g = hg.get_4_adjacency_implicit_graph((6, 7))

        t1, _ = hg.component_tree_max_tree(g, im1)
        t2, _ = hg.component_tree_max_tree(g, im2)

        res = hg.tree_fusion_depth_map((t1, t2))

        expected = np.asarray((0, 0, 0, 0, 0, 0, 0,
                               3, 3, 3, 2, 1, 1, 1,
                               3, 4, 3, 2, 2, 3, 1,
                               3, 3, 3, 2, 2, 3, 1,
                               2, 2, 2, 2, 1, 1, 1,
                               1, 1, 1, 1, 1, 0, 0), dtype=np.float32)

        diff = expected - res
        self.assertTrue(np.all(diff == diff[0]))


if __name__ == '__main__':
    unittest.main()
