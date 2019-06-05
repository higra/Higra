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


class TestAtAccumulator(unittest.TestCase):

    def test_accumulate_at(self):
        indices = np.asarray((1, 1, -1, 2, 0), dtype=np.int64)
        weights = np.asarray((1, 2, 3, 4, 5))

        res = hg.accumulate_at(indices, weights, hg.Accumulators.sum)
        expected_res = np.asarray((5, 3, 4))
        self.assertTrue(np.all(res == expected_res))

        weights_vec = np.asarray(((1, 6),
                                  (2, 7),
                                  (3, 8),
                                  (4, 9),
                                  (5, 10)))
        res_vec = hg.accumulate_at(indices, weights_vec, hg.Accumulators.sum)
        expected_res_vec = np.asarray((
            (5, 10),
            (3, 13),
            (4, 9)))
        self.assertTrue(np.all(res_vec == expected_res_vec))
