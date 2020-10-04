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


class TestRegularGraph(unittest.TestCase):

    @staticmethod
    def graph_implicit_explicit_equal(g1, g2):
        dg1 = set()
        for v1 in g1.vertices():
            for v2 in g1.adjacent_vertices(v1):
                if v2 > v1:
                    dg1.add((v1, v2))

        dg2 = set()
        for v1 in g2.vertices():
            for v2 in g2.adjacent_vertices(v1):
                if v2 > v1:
                    dg2.add((v1, v2))

        return dg1 == dg2

    def test_create_graph(self):
        shape = (2, 3)
        nl = ((-1, 0), (0, -1), (0, 1), (1, 0))
        g1 = hg.RegularGraph2d(hg.EmbeddingGrid2d(shape), nl)

        g2 = hg.get_4_adjacency_implicit_graph(shape)
        g3 = hg.get_8_adjacency_implicit_graph(shape)

        for g in (g1, g2, g3):
            self.assertTrue(g.num_vertices() == 6)

        self.assertTrue(np.all(g1.shape() == shape))
        self.assertTrue(np.all(g1.neighbour_list() == nl))

        g4 = hg.RegularGraph2d(g1.shape(), g1.neighbour_list())
        self.assertTrue(np.all(g4.shape() == shape))
        self.assertTrue(np.all(g4.neighbour_list() == nl))

    def test_dynamic_attributes(self):
        shape = (2, 3)
        g = hg.get_4_adjacency_implicit_graph(shape)
        g.new_attribute = 42
        self.assertTrue(g.new_attribute == 42)

    def test_vertices_iterator(self):
        shape = (2, 3)
        g1 = hg.get_4_adjacency_implicit_graph(shape)
        g2 = hg.get_8_adjacency_implicit_graph(shape)

        vref = [0, 1, 2, 3, 4, 5];

        for g in (g1, g2):
            vtest = [];

            for v in g.vertices():
                vtest.append(v)

            self.assertTrue(vtest == vref)

    def test_out_edge_iterator4(self):
        shape = (2, 3)
        g = hg.get_4_adjacency_implicit_graph(shape)

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
                res.append((g.source(e), g.target(e)))

            self.assertTrue(res == ref[v])

    def test_out_edge_iterator8(self):
        shape = (2, 3)
        g = hg.get_8_adjacency_implicit_graph(shape)

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

    def test_as_explicit_graph(self):
        shape = (2, 3)
        g_imp = hg.get_4_adjacency_implicit_graph(shape)

        g_exp = g_imp.as_explicit_graph()

        self.assertTrue(TestRegularGraph.graph_implicit_explicit_equal(g_imp, g_exp))

        g_imp = hg.get_8_adjacency_implicit_graph(shape)

        g_exp = g_imp.as_explicit_graph()

        self.assertTrue(TestRegularGraph.graph_implicit_explicit_equal(g_imp, g_exp))

    def test_pickle(self):
        import pickle

        tests = ((hg.RegularGraph1d, (3,), ((-1,), (1,))),
                 (hg.RegularGraph2d, (3, 2), ((-1, 0), (1, 0))),
                 (hg.RegularGraph3d, (3, 2, 1), ((-1, 0, -1), (1, 0, 1))),
                 (hg.RegularGraph4d, (3, 2, 1, 2), ((-1, 0, -1, 0), (1, 0, 1, 0))),
                 (hg.RegularGraph5d, (3, 2, 1, 2, 1), ((-1, 0, -1, 0, 1), (1, 0, 1, 0, -1))))

        for c, s, n in tests:
            e = c(s, n)
            hg.set_attribute(e, "test", (1, 2, 3))
            hg.add_tag(e, "foo")

            data = pickle.dumps(e)
            e2 = pickle.loads(data)

            self.assertTrue(np.all(e.shape() == e2.shape()))
            self.assertTrue(np.all(e.neighbour_list() == e2.neighbour_list()))

            self.assertTrue(hg.get_attribute(e, "test") == hg.get_attribute(e2, "test"))
            self.assertTrue(e.test == e2.test)
            self.assertTrue(hg.has_tag(e, "foo"))


if __name__ == '__main__':
    unittest.main()
