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


class TestGraphImage(unittest.TestCase):

    def test_graph_4_adjacency_2_khalimsky(self):
        g = hg.get_4_adjacency_graph((2, 3))
        data = np.asarray((1, 0, 2, 1, 1, 1, 2))

        ref = np.asarray(((0, 1, 0, 2, 0),
                          (0, 1, 1, 2, 1),
                          (0, 1, 0, 2, 0)))
        r = hg.graph_4_adjacency_2_khalimsky(data, g)
        self.assertTrue(np.allclose(ref, r))

    def test_khalimsky_2_graph_4_adjacency(self):
        data = np.asarray((1, 0, 2, 1, 1, 1, 2))

        ref = np.asarray(((0, 1, 0, 2, 0),
                          (0, 1, 1, 2, 1),
                          (0, 1, 0, 2, 0)))
        graph, weights = hg.khalimsky_2_graph_4_adjacency(ref)
        shape = hg.CptGridGraph.get_shape(graph)
        self.assertTrue(np.allclose(shape, (2, 3)))
        self.assertTrue(np.allclose(data, weights))


if __name__ == '__main__':
    unittest.main()
