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

    def test_accumulate_at_indices_below_minus_one(self):
        weights = np.asarray((1, 2, 3))
        weights_vec = np.asarray(((1, 6), (2, 7), (3, 8)))
        for indices in (np.asarray((0, -2, 1), dtype=np.int64),
                        np.asarray((0, -1000000, 1), dtype=np.int32),
                        np.asarray((0, 2 ** 63, 1), dtype=np.uint64)):
            with self.assertRaises(RuntimeError):
                hg.accumulate_at(indices, weights, hg.Accumulators.sum)
            with self.assertRaises(RuntimeError):
                hg.accumulate_at(indices, weights_vec, hg.Accumulators.mean)

    def test_accumulate_at_unreached_rows(self):
        indices = np.asarray((0, 3, 3), dtype=np.int64)
        for weights in (np.asarray((1., 2., 3.)), np.asarray(((1, 6), (2, 7), (3, 8)), dtype=np.int32)):
            info = np.iinfo(weights.dtype) if weights.dtype.kind == 'i' else np.finfo(weights.dtype)
            expected = {
                hg.Accumulators.sum: 0,
                hg.Accumulators.mean: 0,
                hg.Accumulators.prod: 1,
                hg.Accumulators.min: info.max,
                hg.Accumulators.max: info.min,
                hg.Accumulators.first: 0,
                hg.Accumulators.last: 0,
            }
            for acc, value in expected.items():
                with self.subTest(accumulator=acc, shape=weights.shape):
                    res = hg.accumulate_at(indices, weights, acc)
                    self.assertEqual(res.dtype, weights.dtype)
                    np.testing.assert_array_equal(res[1:3], np.full_like(res[1:3], value))

            res = hg.accumulate_at(indices, weights, hg.Accumulators.counter)
            self.assertEqual(res.dtype, np.int64)
            np.testing.assert_array_equal(res, (1, 0, 0, 2))
            res = hg.accumulate_at(indices, weights, hg.Accumulators.argmin)
            self.assertEqual(res.dtype, np.int64)
            np.testing.assert_array_equal(res, (0, -1, -1, 0))
            res = hg.accumulate_at(indices, weights, hg.Accumulators.argmax)
            self.assertEqual(res.dtype, np.int64)
            np.testing.assert_array_equal(res, (0, -1, -1, 1))

    def test_accumulate_at_index_results_small_integer_types(self):
        indices = np.zeros(300, dtype=np.int64)
        weights = np.ones(300, dtype=np.uint8)
        weights[299] = 0
        weights[298] = 2
        np.testing.assert_array_equal(hg.accumulate_at(indices, weights, hg.Accumulators.argmin), (299,))
        np.testing.assert_array_equal(hg.accumulate_at(indices, weights, hg.Accumulators.argmax), (298,))
        np.testing.assert_array_equal(hg.accumulate_at(indices, weights, hg.Accumulators.counter), (300,))


if __name__ == '__main__':
    unittest.main()
