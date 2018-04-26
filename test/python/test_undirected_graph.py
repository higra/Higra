import unittest

import sys

sys.path.insert(0, "@PYTHON_MODULE_PATH@")

import higra as hg
import numpy as np

class TestUndirectedGraph(unittest.TestCase):

    def test_add_vertices(self):
        g = hg.UndirectedGraph()
        self.assertTrue(g.numVertices() == 0)
        self.assertTrue(g.addVertex() == 0)
        self.assertTrue(g.numVertices() == 1)
        self.assertTrue(g.addVertex() == 1)
        self.assertTrue(g.numVertices() == 2)

        g = hg.UndirectedGraph(3)
        self.assertTrue(g.numVertices() == 3)

    def test_add_edges(self):
        g = hg.UndirectedGraph(3)
        self.assertTrue(g.numEdges() == 0)
        g.addEdge(0, 1)
        self.assertTrue(g.numEdges() == 1)

        # parallel edge allowed
        g.addEdge(0, 1)
        self.assertTrue(g.numEdges() == 2)

        # still parallel edge allowed
        g.addEdge(1, 0)
        self.assertTrue(g.numEdges() == 3)

        g.addEdge(0, 2)
        self.assertTrue(g.numEdges() == 4)

    def test_vertex_iterator(self):
        g = hg.getTestUndirectedGraph()
        vref = [0, 1, 2, 3];
        vtest = [];

        for v in g.vertices():
            vtest.append(v)

        self.assertTrue(vtest == vref)

    def test_edge_iterator(self):
        g = hg.getTestUndirectedGraph()
        ref = [(0, 1), (1, 2), (0, 2)]
        test = []

        for e in g.edges():
            test.append(e)

        self.assertTrue(test == ref)

    def test_out_edge_iterator(self):
        g = hg.getTestUndirectedGraph()
        ref = [[(0, 1), (0, 2)],
               [(1, 0), (1, 2)],
               [(2, 1), (2, 0)],
               []]
        test = []
        for v in g.vertices():
            test.append([])
            for e in g.outEdges(v):
                test[v].append(e)

        self.assertTrue(test == ref)

    def test_in_edge_iterator(self):
        g = hg.getTestUndirectedGraph()
        ref = [[(1, 0), (2, 0)],
               [(0, 1), (2, 1)],
               [(1, 2), (0, 2)],
               []]
        test = []
        for v in g.vertices():
            test.append([])
            for e in g.inEdges(v):
                test[v].append(e)

        self.assertTrue(test == ref)

    def test_adjacent_vertex_iterator(self):
        g = hg.getTestUndirectedGraph()
        ref = [[1, 2],
               [0, 2],
               [1, 0],
               []]
        test = []
        for v in g.vertices():
            test.append([])
            for av in g.adjacentVertices(v):
                test[v].append(av)

        self.assertTrue(test == ref)

    def test_degrees(self):
        g = hg.getTestUndirectedGraph()
        self.assertTrue(g.degree(0) == 2)
        self.assertTrue(g.outDegree(0) == 2)
        self.assertTrue(g.inDegree(0) == 2)

        self.assertTrue(g.degree(1) == 2)
        self.assertTrue(g.outDegree(1) == 2)
        self.assertTrue(g.inDegree(1) == 2)

        self.assertTrue(g.degree(2) == 2)
        self.assertTrue(g.outDegree(2) == 2)
        self.assertTrue(g.inDegree(2) == 2)

        self.assertTrue(g.degree(3) == 0)
        self.assertTrue(g.outDegree(3) == 0)
        self.assertTrue(g.inDegree(3) == 0)

        indices = np.asarray(((0, 3), (1, 2)))
        ref = np.asarray(((2, 0), (2, 2)))
        self.assertTrue(np.allclose(g.degree(indices), ref))
        self.assertTrue(np.allclose(g.inDegree(indices), ref))
        self.assertTrue(np.allclose(g.outDegree(indices), ref))

    def test_edge_index_iterator(self):
        g = hg.getTestUndirectedGraph()
        ref = [0, 1, 2]
        test = []

        for e in g.edgeIndexes():
            test.append(e)

        self.assertTrue(test == ref)

    def test_out_edge_index_iterator(self):
        g = hg.getTestUndirectedGraph()
        ref = [[0, 2],
               [0, 1],
               [1, 2],
               []]
        test = []
        for v in g.vertices():
            test.append([])
            for e in g.outEdgeIndexes(v):
                test[v].append(e)

        self.assertTrue(test == ref)

    def test_in_edge_index_iterator(self):
        g = hg.getTestUndirectedGraph()
        ref = [[0, 2],
               [0, 1],
               [1, 2],
               []]
        test = []
        for v in g.vertices():
            test.append([])
            for e in g.inEdgeIndexes(v):
                test[v].append(e)

        self.assertTrue(test == ref)


if __name__ == '__main__':
    unittest.main()
