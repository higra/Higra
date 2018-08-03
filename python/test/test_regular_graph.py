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


class TestRegularGraph(unittest.TestCase):

    def test_create_graph(self):
        shape = (2, 3)
        nl = ((-1, 0), (0, -1), (0, 1), (1, 0))
        g1 = hg.RegularGraph2d(hg.EmbeddingGrid2d(shape), nl)

        g2 = hg._get_4_adjacency_implicit_graph(shape)
        g3 = hg._get_8_adjacency_implicit_graph(shape)

        for g in (g1, g2, g3):
            self.assertTrue(g.num_vertices() == 6)

    def test_vertices_iterator(self):
        shape = (2, 3)
        g1 = hg._get_4_adjacency_implicit_graph(shape)
        g2 = hg._get_8_adjacency_implicit_graph(shape)

        vref = [0, 1, 2, 3, 4, 5];

        for g in (g1, g2):
            vtest = [];

            for v in g.vertices():
                vtest.append(v)

            self.assertTrue(vtest == vref)

    def test_out_edge_iterator4(self):
        shape = (2, 3)
        g = hg._get_4_adjacency_implicit_graph(shape)

        ref = [[(0, 1), (0, 3)],
               [(1, 0), (1, 2), (1, 4)],
               [(2, 1), (2, 5)],
               [(3, 0), (3, 4)],
               [(4, 1), (4, 3), (4, 5)],
               [(5, 2), (5, 4)]
               ]

        for v in g.vertices():
            res = []
            for e in g.out_edges(v):
                res.append(e)
            self.assertTrue(res == ref[v])

    def test_out_edge_iterator8(self):
        shape = (2, 3)
        g = hg._get_8_adjacency_implicit_graph(shape)

        ref = [[(0, 1), (0, 3), (0, 4)],
               [(1, 0), (1, 2), (1, 3), (1, 4), (1, 5)],
               [(2, 1), (2, 4), (2, 5)],
               [(3, 0), (3, 1), (3, 4)],
               [(4, 0), (4, 1), (4, 2), (4, 3), (4, 5)],
               [(5, 1), (5, 2), (5, 4)]
               ]

        for v in g.vertices():
            res = []
            for e in g.out_edges(v):
                res.append(e)
            self.assertTrue(res == ref[v])


if __name__ == '__main__':
    unittest.main()
