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
import higra as hg
import numpy as np


class TestSorting(unittest.TestCase):

    def test_sort(self):
        a = np.asarray((5, 2, 1, 4, 9))
        hg.sort(a)
        self.assertTrue(np.all(a == (1, 2, 4, 5, 9)))

        a = np.asarray((5, 2, 1, 4, 9))
        hg.sort(a, stable=True)
        self.assertTrue(np.all(a == (1, 2, 4, 5, 9)))

    def test_arg_sort(self):
        a = np.asarray((5, 2, 1, 4, 9))
        i = hg.arg_sort(a)
        self.assertTrue(np.all(i == (2, 1, 3, 0 , 4)))

        a = np.asarray((2, 2, 2, 2, 1, 1, 1, 1))
        i = hg.arg_sort(a, stable=True)
        self.assertTrue(np.all(i == (4, 5, 6, 7, 0, 1, 2, 3)))

        a = np.asarray(((2, 2, 1, 1, 3),
                        (2, 2, 2, 1, 0))).T
        i = hg.arg_sort(a, stable=True)
        self.assertTrue(np.all(i == (3, 2, 0, 1, 4)))
