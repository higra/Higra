############################################################################
# Copyright ESIEE Paris (2023)                                             #
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


class TestBipartiteGraph(unittest.TestCase):

    def test_is_bipartite_graph_empty(self):
        g = hg.UndirectedGraph(0)

        ans, color = hg.is_bipartite_graph(g)

        self.assertTrue(ans)
        self.assertTrue(len(color) == 0)

        ans, color = hg.is_bipartite_graph((g.sources(), g.targets(), g.num_vertices()))

        self.assertTrue(ans)
        self.assertTrue(len(color) == 0)

    def test_is_bipartite_graph_true(self):
        g = hg.UndirectedGraph(6)
        g.add_edges(np.array([0, 0, 4, 2]), np.array([1, 4, 3, 3]))

        ans, color = hg.is_bipartite_graph(g)

        self.assertTrue(ans)
        self.assertTrue(len(color) == 6)
        self.assertTrue(np.all(color[g.sources()] != color[g.targets()]))

        ans, color = hg.is_bipartite_graph((g.sources(), g.targets(), g.num_vertices()))

        self.assertTrue(ans)
        self.assertTrue(len(color) == 6)
        self.assertTrue(np.all(color[g.sources()] != color[g.targets()]))

    def test_is_bipartite_graph_false(self):
        g = hg.UndirectedGraph(6)
        g.add_edges(np.array([0, 0, 1, 1, 2, 1, 5]), np.array([3, 4, 3, 5, 5, 4, 4]))

        ans, color = hg.is_bipartite_graph(g)

        self.assertFalse(ans)
        self.assertTrue(len(color) == 0)

        ans, color = hg.is_bipartite_graph((g.sources(), g.targets(), g.num_vertices()))

        self.assertFalse(ans)
        self.assertTrue(len(color) == 0)
