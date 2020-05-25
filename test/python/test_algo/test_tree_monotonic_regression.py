############################################################################
# Copyright ESIEE Paris (2020)                                             #
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


class TestTreeMonotonicRegression(unittest.TestCase):

    def test_tree_monotonic_regression_trivial(self):
        tree = hg.Tree((7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11))
        altitudes = np.asarray((0, 1, 0, 2, 0, 0, 0, 2, 3, 0, 5, 10))

        res = hg.tree_monotonic_regression(tree, altitudes, "max")
        self.assertTrue(np.all(altitudes == res))

        res = hg.tree_monotonic_regression(tree, altitudes, "min")
        self.assertTrue(np.all(altitudes == res))

        res = hg.tree_monotonic_regression(tree, altitudes, "least_square")
        self.assertTrue(np.all(altitudes == res))

        res = hg.tree_monotonic_regression(tree, altitudes, "max")
        self.assertTrue(np.all(altitudes == res))

        with self.assertRaises(Exception):
            hg.tree_monotonic_regression(tree, altitudes, "truc")

    def test_tree_monotonic_regression_max(self):
        tree = hg.Tree((7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11))
        altitudes = np.asarray((0, 3, 0, 2, 0, 0, 0, 2, 3, 0, 5, 4))

        ref = np.asarray((0, 3, 0, 2, 0, 0, 0, 3, 3, 0, 5, 5))
        res = hg.tree_monotonic_regression(tree, altitudes, "max")
        self.assertTrue(np.all(ref == res))

    def test_tree_monotonic_regression_min(self):
        tree = hg.Tree((7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11))
        altitudes = np.asarray((0, 3, 0, 2, 0, 0, 0, 2, 3, 0, 5, 4))

        ref = np.asarray((0, 2, 0, 2, 0, 0, 0, 2, 3, 0, 4, 4))
        res = hg.tree_monotonic_regression(tree, altitudes, "min")
        self.assertTrue(np.all(ref == res))

    def test_tree_monotonic_regression_least_square(self):
        tree = hg.Tree((5, 5, 6, 6, 7, 7, 7, 7))
        altitudes = np.asarray((13, 14, 6, 8, 7, 11, 5, 10))

        ref = np.asarray((12, 12, 6, 6.5, 7, 12, 6.5, 12))
        res = hg.tree_monotonic_regression(tree, altitudes, "least_square")
        self.assertTrue(np.allclose(ref, res))

        weights = np.asarray((1, 1, 1, 1, 1, 1, 2, 1))
        ref = np.asarray((12, 12, 6, 6, 7, 12, 6, 12))
        res = hg.tree_monotonic_regression(tree, altitudes, "least_square", weights)
        self.assertTrue(np.allclose(ref, res))
