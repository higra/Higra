import unittest

import sys

sys.path.insert(0, "@PYTHON_MODULE_PATH@")

import higra as hg
import numpy as np


class TestUndirectedGraph(unittest.TestCase):

    @staticmethod
    def test_graph():
        g = hg.UndirectedGraph(4)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(0, 2)
        return g


    def test_add_vertices(self):
        g = hg.UndirectedGraph()
        self.assertTrue(g.num_vertices() == 0)
        self.assertTrue(g.add_vertex() == 0)
        self.assertTrue(g.num_vertices() == 1)
        self.assertTrue(g.add_vertex() == 1)
        self.assertTrue(g.num_vertices() == 2)

        g = hg.UndirectedGraph(3)
        self.assertTrue(g.num_vertices() == 3)

    def test_add_edges(self):
        g = hg.UndirectedGraph(3)
        self.assertTrue(g.num_edges() == 0)
        g.add_edge(0, 1)
        self.assertTrue(g.num_edges() == 1)

        # parallel edge allowed
        g.add_edge(0, 1)
        self.assertTrue(g.num_edges() == 2)

        # still parallel edge allowed
        g.add_edge(1, 0)
        self.assertTrue(g.num_edges() == 3)

        g.add_edge(0, 2)
        self.assertTrue(g.num_edges() == 4)

    def test_vertex_iterator(self):
        g = TestUndirectedGraph.test_graph()
        vref = [0, 1, 2, 3];
        vtest = [];

        for v in g.vertices():
            vtest.append(v)

        self.assertTrue(vtest == vref)

    def test_edge_iterator(self):
        g = TestUndirectedGraph.test_graph()
        ref = [(0, 1), (1, 2), (0, 2)]
        test = []

        for e in g.edges():
            test.append(e)

        self.assertTrue(test == ref)

    def test_out_edge_iterator(self):
        g = TestUndirectedGraph.test_graph()
        ref = [[(0, 1), (0, 2)],
               [(1, 0), (1, 2)],
               [(2, 1), (2, 0)],
               []]
        test = []
        for v in g.vertices():
            test.append([])
            for e in g.out_edges(v):
                test[v].append(e)

        self.assertTrue(test == ref)

    def test_in_edge_iterator(self):
        g = TestUndirectedGraph.test_graph()
        ref = [[(1, 0), (2, 0)],
               [(0, 1), (2, 1)],
               [(1, 2), (0, 2)],
               []]
        test = []
        for v in g.vertices():
            test.append([])
            for e in g.in_edges(v):
                test[v].append(e)

        self.assertTrue(test == ref)

    def test_adjacent_vertex_iterator(self):
        g = TestUndirectedGraph.test_graph()
        ref = [[1, 2],
               [0, 2],
               [1, 0],
               []]
        test = []
        for v in g.vertices():
            test.append([])
            for av in g.adjacent_vertices(v):
                test[v].append(av)

        self.assertTrue(test == ref)

    def test_degrees(self):
        g = TestUndirectedGraph.test_graph()
        self.assertTrue(g.degree(0) == 2)
        self.assertTrue(g.out_degree(0) == 2)
        self.assertTrue(g.in_degree(0) == 2)

        self.assertTrue(g.degree(1) == 2)
        self.assertTrue(g.out_degree(1) == 2)
        self.assertTrue(g.in_degree(1) == 2)

        self.assertTrue(g.degree(2) == 2)
        self.assertTrue(g.out_degree(2) == 2)
        self.assertTrue(g.in_degree(2) == 2)

        self.assertTrue(g.degree(3) == 0)
        self.assertTrue(g.out_degree(3) == 0)
        self.assertTrue(g.in_degree(3) == 0)

        indices = np.asarray(((0, 3), (1, 2)))
        ref = np.asarray(((2, 0), (2, 2)))
        self.assertTrue(np.allclose(g.degree(indices), ref))
        self.assertTrue(np.allclose(g.in_degree(indices), ref))
        self.assertTrue(np.allclose(g.out_degree(indices), ref))

    def test_edge_index_iterator(self):
        g = TestUndirectedGraph.test_graph()
        ref = [0, 1, 2]
        test = []

        for e in g.edge_index_iterator():
            test.append(e)

        self.assertTrue(test == ref)

    def test_out_edge_index_iterator(self):
        g = TestUndirectedGraph.test_graph()
        ref = [[0, 2],
               [0, 1],
               [1, 2],
               []]
        test = []
        for v in g.vertices():
            test.append([])
            for e in g.out_edge_index_iterator(v):
                test[v].append(e)

        self.assertTrue(test == ref)

    def test_in_edge_index_iterator(self):
        g = TestUndirectedGraph.test_graph()
        ref = [[0, 2],
               [0, 1],
               [1, 2],
               []]
        test = []
        for v in g.vertices():
            test.append([])
            for e in g.in_edge_index_iterator(v):
                test[v].append(e)

        self.assertTrue(test == ref)


if __name__ == '__main__':
    unittest.main()
