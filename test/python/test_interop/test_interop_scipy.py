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


class TestSciPyInterop(unittest.TestCase):

    def test_binary_hierarchy_to_scipy_linkage_matrix(self):
        t = hg.Tree((5, 5, 7, 6, 6, 7, 8, 8, 8))
        ref = np.asarray(((0, 1, 1 / 3, 2),
                          (3, 4, 2 / 3, 2),
                          (2, 5, 2 / 3, 3),
                          (6, 7, 1, 5)))
        res = hg.binary_hierarchy_to_scipy_linkage_matrix(t)
        self.assertTrue(np.allclose(ref, res))

    def test_scipy_linkage_matrix_to_binary_hierarchy(self):
        linkage_matrix = np.asarray(((0, 1, 1, 2),
                                     (3, 4, 2, 2),
                                     (2, 5, 2, 3),
                                     (6, 7, 3, 5)))
        t, altitudes, area = hg.scipy_linkage_matrix_to_binary_hierarchy(linkage_matrix)

        self.assertTrue(np.all(t.parents() == (5, 5, 7, 6, 6, 7, 8, 8, 8)))
        self.assertTrue(np.all(altitudes == (0, 0, 0, 0, 0, 1, 2, 2, 3)))
        self.assertTrue(np.all(area == (1, 1, 1, 1, 1, 2, 2, 3, 5)))
