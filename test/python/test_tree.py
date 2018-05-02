import unittest
import numpy as np
import sys

sys.path.insert(0, "@PYTHON_MODULE_PATH@")

import higra as hg


class TestTree(unittest.TestCase):

    @staticmethod
    def get_tree():
        parent_relation = np.asarray((5, 5, 6, 6, 6, 7, 7, 7), dtype=np.uint64)
        return hg.Tree(parent_relation)

    def test_size_tree(self):
        t = TestTree.get_tree()

        self.assertTrue(t.root() == 7)
        self.assertTrue(t.num_vertices() == 8)
        self.assertTrue(t.num_edges() == 7)
        self.assertTrue(t.num_leaves() == 5)

    def test_vertex_iterator(self):
        t = TestTree.get_tree()

        ref = [0, 1, 2, 3, 4, 5, 6, 7];
        res = []

        for v in t.vertices():
            res.append(v)
        self.assertTrue(res == ref)

    def test_tree_degree(self):
        t = TestTree.get_tree()

        ref = [1, 1, 1, 1, 1, 3, 4, 2]

        for v in t.vertices():
            self.assertTrue(t.degree(v) == ref[v])
            self.assertTrue(t.in_degree(v) == ref[v])
            self.assertTrue(t.out_degree(v) == ref[v])

    def test_ctr_fail(self):
        with self.assertRaises(RuntimeError):
            hg.Tree((5, 0, 6, 6, 6, 7, 7, 7))
        with self.assertRaises(RuntimeError):
            hg.Tree((5, 1, 6, 6, 6, 7, 7, 7))
        with self.assertRaises(RuntimeError):
            hg.Tree((5, 1, 6, 6, 6, 7, 7, 2))
        with self.assertRaises(RuntimeError):
            hg.Tree((2, 2, 4, 4, 4))

    def test_edge_iterator(self):
        t = TestTree.get_tree()

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

    def test_adjacent_vertex_iterator(self):
        t = TestTree.get_tree()

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
            for a in t.adjacent_vertices(v):
                res.append(a)
            self.assertTrue(res == ref[v])

    def test_out_edge_iterator(self):
        t = TestTree.get_tree()

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
            for e in t.out_edges(v):
                res.append(e)
            self.assertTrue(res == ref[v])

    def test_in_edge_iterator(self):
        t = TestTree.get_tree()

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
            for e in t.in_edges(v):
                res.append(e)
            self.assertTrue(res == ref[v])

    def test_edge_index_iterator(self):
        t = TestTree.get_tree()

        ref = [0, 1, 2, 3, 4, 5, 6]
        res = []

        for ei in t.edge_indexes():
            res.append(ei)

        self.assertTrue(res == ref)

    def test_out_edge_index_iterator(self):
        t = TestTree.get_tree()

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
            for ei in t.out_edge_indexes(v):
                res.append(ei)
            self.assertTrue(res == ref[v])

    def test_in_edge_index_iterator(self):
        t = TestTree.get_tree()

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
            for ei in t.in_edge_indexes(v):
                res.append(ei)
            self.assertTrue(res == ref[v])

    def test_num_children(self):
        t = TestTree.get_tree()

        ref = [0, 0, 0, 0, 0, 2, 3, 2]
        res = []

        for v in t.vertices():
            res.append(t.num_children(v))
        self.assertTrue(res == ref)

    def test_children_iterator(self):
        t = TestTree.get_tree()

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
