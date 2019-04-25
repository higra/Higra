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


class TestHGUtils(unittest.TestCase):

    def test_linearize_vertex_weights(self):
        g = hg.get_4_adjacency_graph((4, 5))

        r = hg.linearize_vertex_weights(np.ones((20,)), g, (4, 5))
        self.assertTrue(r.shape == (20,))

        r = hg.linearize_vertex_weights(np.ones((4, 5)), g, (4, 5))
        self.assertTrue(r.shape == (20,))

        r = hg.linearize_vertex_weights(np.ones((4, 5, 10, 12)), g, (4, 5))
        self.assertTrue(r.shape == (20, 10, 12))

        r = hg.linearize_vertex_weights(np.ones((20, 4, 5, 2, 3)), g, (4, 5))
        self.assertTrue(r.shape == (20, 4, 5, 2, 3))

        with self.assertRaises(Exception):
            hg.linearize_vertex_weights(np.ones((5, 4)), g, (4, 5))

        with self.assertRaises(Exception):
            hg.linearize_vertex_weights(np.ones((25,)), g, (4, 5))

    def test_delinearize_vertex_weights(self):
        g = hg.get_4_adjacency_graph((4, 5))

        r = hg.delinearize_vertex_weights(np.ones((20,)), g, (4, 5))
        self.assertTrue(r.shape == (4, 5))

        r = hg.delinearize_vertex_weights(np.ones((4, 5)), g, (4, 5))
        self.assertTrue(r.shape == (4, 5))

        r = hg.delinearize_vertex_weights(np.ones((4, 5, 10, 12)), g, (4, 5))
        self.assertTrue(r.shape == (4, 5, 10, 12))

        r = hg.delinearize_vertex_weights(np.ones((20, 4, 5, 2, 3)), g, (4, 5))
        self.assertTrue(r.shape == (4, 5, 4, 5, 2, 3))

        with self.assertRaises(Exception):
            hg.delinearize_vertex_weights(np.ones((5, 4)), g, (4, 5))

        with self.assertRaises(Exception):
            hg.delinearize_vertex_weights(np.ones((25,)), g, (4, 5))

    def test_common_type(self):
        a_uint16 = np.zeros((1, 1), dtype=np.uint16)
        a_int8 = np.zeros((1, 1), dtype=np.int8)
        a_uint64 = np.zeros((1, 1), dtype=np.uint64)

        self.assertTrue(hg.common_type(a_uint16, a_int8, a_uint64) == np.int64)
        self.assertTrue(hg.common_type(a_uint16, a_int8) == np.int32)
        self.assertTrue(hg.common_type(a_uint16, a_uint16) == np.uint16)

        self.assertTrue(hg.common_type(a_uint16, a_int8, a_uint64, safety_level='overflow') == np.float64)
        self.assertTrue(hg.common_type(a_uint16, a_int8, safety_level='overflow') == np.float64)
        self.assertTrue(hg.common_type(a_uint16, a_uint16, safety_level='overflow') == np.float64)

    def cast_to_common_type(self):
        a_uint16 = np.zeros((1, 1), dtype=np.uint16)
        a_int8 = np.zeros((1, 1), dtype=np.int8)
        a_uint64 = np.zeros((1, 1), dtype=np.uint64)
        a_int64 = np.zeros((1, 1), dtype=np.int64)

        a, b, c, d = hg.common_type(a_uint16, a_int8, a_int64, a_uint64)
        self.assertTrue(a.dtype == np.int64)
        self.assertTrue(b.dtype == np.int64)
        self.assertTrue(c.dtype == np.int64)
        self.assertTrue(d.dtype == np.int64)

        self.assertTrue(id(c) == id(a_int64))
