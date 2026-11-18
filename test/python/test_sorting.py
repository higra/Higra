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

    def test_sort_non_contiguous(self):
        for dtype in (np.int8, np.uint16, np.int32, np.int64, np.uint64, np.float32, np.float64):
            for stable in (False, True):
                values = np.asarray((5, 2, 1, 4, 9, 3, 7, 0), dtype=dtype)

                parent = values.copy()
                hg.sort(parent[::2], stable=stable)
                self.assertTrue(np.array_equal(parent, np.asarray((1, 2, 5, 4, 7, 3, 9, 0), dtype=dtype)))

                parent = values.copy()
                hg.sort(parent[::-1], stable=stable)
                self.assertTrue(np.array_equal(parent, np.asarray((9, 7, 5, 4, 3, 2, 1, 0), dtype=dtype)))

                parent = np.stack([values, values[::-1]], axis=1)
                hg.sort(parent[:, 0], stable=stable)
                self.assertTrue(np.array_equal(parent[:, 0], np.sort(values)))
                self.assertTrue(np.array_equal(parent[:, 1], values[::-1]))

                parent = np.asfortranarray(np.stack([values, values[::-1]], axis=0))
                hg.sort(parent[1], stable=stable)
                self.assertTrue(np.array_equal(parent[0], values))
                self.assertTrue(np.array_equal(parent[1], np.sort(values)))

    def test_sort_unaligned(self):
        buffer = np.zeros(8 * 6 + 1, dtype=np.uint8)
        a = buffer[1:].view(np.int64)
        self.assertFalse(a.flags.aligned)
        a[:] = (5, 2, 1, 4, 9, 0)
        hg.sort(a)
        self.assertTrue(np.array_equal(a, (0, 1, 2, 4, 5, 9)))

    def test_sort_stable_non_contiguous(self):
        # +0. and -0. compare equal: a stable sort keeps their order in the (reversed) view
        parent = np.asarray((0., 1, -0., 1, -1))
        a = parent[::-1]
        hg.sort(a, stable=True)
        self.assertTrue(np.array_equal(a, (-1, 0, 0, 1, 1)))
        self.assertTrue(np.array_equal(np.signbit(a), (True, True, False, False, False)))
        self.assertTrue(np.array_equal(np.signbit(parent), (False, False, False, True, True)))

    def test_sort_read_only(self):
        a = np.asarray((5, 2, 1, 4, 9))
        a.flags.writeable = False
        with self.assertRaises(ValueError):
            hg.sort(a)
        self.assertTrue(np.array_equal(a, (5, 2, 1, 4, 9)))
        with self.assertRaises(ValueError):
            hg.sort(a[::2], stable=True)
        self.assertTrue(np.array_equal(a, (5, 2, 1, 4, 9)))

    def test_sort_unsupported_inputs(self):
        # other dtypes, non-native byte order and non-arrays raise TypeError and are not modified
        for a in (np.asarray((5, 2, 1, 4)).astype('>i8'), np.asarray((5., 2., 1.), np.float16),
                  np.asarray((True, False, True)), np.asarray((3 + 1j, 1 + 0j)), np.asarray(("b", "a"))):
            for stable in (False, True):
                before = a.copy()
                with self.assertRaises(TypeError):
                    hg.sort(a, stable=stable)
                self.assertTrue(np.array_equal(a, before))
                self.assertEqual(a.dtype, before.dtype)
        a = [5, 2, 1]
        with self.assertRaises(TypeError):
            hg.sort(a)
        self.assertEqual(a, [5, 2, 1])
        with self.assertRaises(RuntimeError):
            hg.sort(np.zeros((2, 2)))

    def test_arg_sort_non_contiguous(self):
        a = np.asarray((5, 2, 1, 4, 9, 3, 7, 0))
        for dtype in (np.uint8, np.int32, np.float32, np.float64):
            x = a.astype(dtype)
            expected = hg.arg_sort(x, stable=True)
            for v in (np.ascontiguousarray(x[::-1])[::-1], np.stack([x, x], 1)[:, 0]):
                self.assertTrue(np.array_equal(hg.arg_sort(v, stable=True), expected))
                self.assertTrue(np.array_equal(hg.arg_sort(v), np.argsort(x, kind="stable")))
            read_only = x.copy()
            read_only.flags.writeable = False
            self.assertTrue(np.array_equal(hg.arg_sort(read_only[::2]), hg.arg_sort(x[::2].copy())))

        b = np.asarray(((2, 2, 1, 1, 3),
                        (2, 2, 2, 1, 0)), dtype=np.float32)
        for v in (b.T, np.asfortranarray(b.T), np.ascontiguousarray(b.T)):
            self.assertTrue(np.array_equal(hg.arg_sort(v, stable=True), (3, 2, 0, 1, 4)))
