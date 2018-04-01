import unittest
import numpy as np
import sys

sys.path.insert(0, "@PYTHON_MODULE_PATH@")

import higra as hg


class TestTree(unittest.TestCase):

    @staticmethod
    def getTree():
        parentRelation = np.asarray((5, 5, 6, 6, 6, 7, 7, 7), dtype=np.uint64)
        return hg.Tree(parentRelation)

    def test_sizeTree(self):
        t = TestTree.getTree()

        self.assertTrue(t.root() == 7)
        self.assertTrue(t.numVertices() == 8)
        self.assertTrue(t.numEdges() == 7)
        self.assertTrue(t.numLeaves() == 5)

    def test_vertexIterator(self):
        t = TestTree.getTree()

        ref = [0, 1, 2, 3, 4, 5, 6, 7];
        res = []

        for v in t.vertices():
            res.append(v)
        self.assertTrue(res == ref)

    def test_treeDegree(self):
        t = TestTree.getTree()

        ref = [1, 1, 1, 1, 1, 3, 4, 2]

        for v in t.vertices():
            self.assertTrue(t.degree(v) == ref[v])
            self.assertTrue(t.inDegree(v) == ref[v])
            self.assertTrue(t.outDegree(v) == ref[v])

    def test_ctrFail(self):
        with self.assertRaises(RuntimeError):
            hg.Tree((5, 0, 6, 6, 6, 7, 7, 7))
        with self.assertRaises(RuntimeError):
            hg.Tree((5, 1, 6, 6, 6, 7, 7, 7))
        with self.assertRaises(RuntimeError):
            hg.Tree((5, 1, 6, 6, 6, 7, 7, 2))
        with self.assertRaises(RuntimeError):
            hg.Tree((2, 2, 4, 4, 4))

    def test_edgeIterator(self):
        t = TestTree.getTree()

        ref = [(0, 5),
               (1, 5),
               (2, 6),
               (3, 6),
               (4, 6),
               (5, 7),
               (6, 7)]
        res = []

        for e in t.edges():
            res.append(e)

        self.assertTrue(res == ref)

    def test_adjacentVertexIterator(self):
        t = TestTree.getTree()

        ref = [[5],
               [5],
               [6],
               [6],
               [6],
               [7, 0, 1],
               [7, 2, 3, 4],
               [5, 6]]

        for v in t.vertices():
            res = []
            for a in t.adjacentVertices(v):
                res.append(a)
            self.assertTrue(res == ref[v])

    def test_outEdgeIterator(self):
        t = TestTree.getTree()

        ref = [[(0, 5)],
               [(1, 5)],
               [(2, 6)],
               [(3, 6)],
               [(4, 6)],
               [(5, 7), (5, 0), (5, 1)],
               [(6, 7), (6, 2), (6, 3), (6, 4)],
               [(7, 5), (7, 6)]];
        for v in t.vertices():
            res = []
            for e in t.outEdges(v):
                res.append(e)
            self.assertTrue(res == ref[v])

    def test_inEdgeIterator(self):
        t = TestTree.getTree()

        ref = [[(5, 0)],
               [(5, 1)],
               [(6, 2)],
               [(6, 3)],
               [(6, 4)],
               [(7, 5), (0, 5), (1, 5)],
               [(7, 6), (2, 6), (3, 6), (4, 6)],
               [(5, 7), (6, 7)]];
        for v in t.vertices():
            res = []
            for e in t.inEdges(v):
                res.append(e)
            self.assertTrue(res == ref[v])

    def test_edgeIndexIterator(self):
        t = TestTree.getTree()

        ref = [0, 1, 2, 3, 4, 5, 6]
        res = []

        for ei in t.edgeIndexes():
            res.append(ei)

        self.assertTrue(res == ref)

    def test_outEdgeIndexIterator(self):
        t = TestTree.getTree()

        ref = [[0],
               [1],
               [2],
               [3],
               [4],
               [5, 0, 1],
               [6, 2, 3, 4],
               [5, 6]]

        for v in t.vertices():
            res = []
            for ei in t.outEdgeIndexes(v):
                res.append(ei)
            self.assertTrue(res == ref[v])

    def test_inEdgeIndexIterator(self):
        t = TestTree.getTree()

        ref = [[0],
               [1],
               [2],
               [3],
               [4],
               [5, 0, 1],
               [6, 2, 3, 4],
               [5, 6]]

        for v in t.vertices():
            res = []
            for ei in t.inEdgeIndexes(v):
                res.append(ei)
            self.assertTrue(res == ref[v])

    def test_numChildren(self):
        t = TestTree.getTree()

        ref = [0, 0, 0, 0, 0, 2, 3, 2]
        res = []

        for v in t.vertices():
            res.append(t.numChildren(v))
        self.assertTrue(res == ref)

    def test_childrenIterator(self):
        t = TestTree.getTree()

        ref = [[],
               [],
               [],
               [],
               [],
               [0, 1],
               [2, 3, 4],
               [5, 6]]

        for v in t.vertices():
            res = []
            for c in t.children(v):
                res.append(c)
            self.assertTrue(res == ref[v])


if __name__ == '__main__':
    unittest.main()
