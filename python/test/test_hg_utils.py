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

        r = hg.linearize_vertex_weights(np.ones((20, )), g, (4, 5))
        self.assertTrue(r.shape == (20, ))

        r = hg.linearize_vertex_weights(np.ones((4, 5)), g, (4, 5))
        self.assertTrue(r.shape == (20, ))

        r = hg.linearize_vertex_weights(np.ones((4, 5, 10, 12)), g, (4, 5))
        self.assertTrue(r.shape == (20, 10, 12))

        r = hg.linearize_vertex_weights(np.ones((20, 4, 5, 10, 12)), g, (4, 5))
        self.assertTrue(r.shape == (20, 4, 5, 10, 12))

        with self.assertRaises(Exception):
            hg.linearize_vertex_weights(np.ones((5, 4)), g, (4, 5))

        with self.assertRaises(Exception):
            hg.linearize_vertex_weights(np.ones((25, )), g, (4, 5))
