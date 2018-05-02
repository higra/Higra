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

    def test_tree_accumulator(self):
        tree = TestTree.get_tree()
        input_array = np.asarray((1, 1, 1, 1, 1, 1, 1, 1))

        res1 = tree.accumulate_parallel(input_array, hg.Accumulators.sum)
        ref1 = np.asarray((0, 0, 0, 0, 0, 2, 3, 2))
        self.assertTrue(np.allclose(ref1, res1))

        leaf_data = np.asarray((1, 1, 1, 1, 1))
        res2 = tree.accumulate_sequential(leaf_data, hg.Accumulators.sum)
        ref2 = np.asarray((1, 1, 1, 1, 1, 2, 3, 5))
        self.assertTrue(np.allclose(ref2, res2))

        res3 = tree.accumulate_and_add_sequential(input_array, leaf_data, hg.Accumulators.max)
        ref3 = np.asarray((1, 1, 1, 1, 1, 2, 2, 3))
        self.assertTrue(np.allclose(ref3, res3))

        input_array = np.asarray((1, 2, 1, 2, 1, 1, 4, 5))
        res4 = tree.accumulate_and_max_sequential(input_array, leaf_data, hg.Accumulators.sum)
        ref4 = np.asarray((1, 1, 1, 1, 1, 2, 4, 6))
        self.assertTrue(np.allclose(ref4, res4))

        input_array = np.asarray((1, 2, 1, 2, 1, 2, 3, 1))
        res5 = tree.accumulate_and_multiply_sequential(input_array, leaf_data, hg.Accumulators.sum)
        ref5 = np.asarray((1, 1, 1, 1, 1, 4, 9, 13))
        self.assertTrue(np.allclose(ref5, res5))

        input_array = np.asarray((1, 2, 1, 2, 1, 4, 2, 10))
        res6 = tree.accumulate_and_min_sequential(input_array, leaf_data, hg.Accumulators.sum)
        ref6 = np.asarray((1, 1, 1, 1, 1, 2, 2, 4))
        self.assertTrue(np.allclose(ref6, res6))

    def test_tree_accumulatorVec(self):
        tree = TestTree.get_tree()
        input_array = np.asarray(((1, 0),
                            (1, 1),
                            (1, 2),
                            (1, 3),
                            (1, 4),
                            (1, 5),
                            (1, 6),
                            (1, 7)))

        res1 = tree.accumulate_parallel(input_array, hg.Accumulators.sum)
        ref1 = np.asarray(((0, 0),
                           (0, 0),
                           (0, 0),
                           (0, 0),
                           (0, 0),
                           (2, 1),
                           (3, 9),
                           (2, 11)))
        self.assertTrue(np.allclose(ref1, res1))

        leaf_data = np.asarray(((1, 0),
                                (1, 1),
                                (1, 2),
                                (1, 3),
                                (1, 4)))
        res2 = tree.accumulate_sequential(leaf_data, hg.Accumulators.sum)
        ref2 = np.asarray(((1, 0),
                           (1, 1),
                           (1, 2),
                           (1, 3),
                           (1, 4),
                           (2, 1),
                           (3, 9),
                           (5, 10)))
        self.assertTrue(np.allclose(ref2, res2))

        res3 = tree.accumulate_and_add_sequential(input_array, leaf_data, hg.Accumulators.sum)
        ref3 = np.asarray(((1, 0),
                           (1, 1),
                           (1, 2),
                           (1, 3),
                           (1, 4),
                           (3, 6),
                           (4, 15),
                           (8, 28)))
        self.assertTrue(np.allclose(ref3, res3))

    def test_tree_propagate(self):
        tree = TestTree.get_tree()
        input_array = np.asarray(((1, 8), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (8, 1)), dtype=np.float64)
        condition = np.asarray((True, False, True, False, True, True, False, False))

        output = tree.propagate_parallel(input_array, condition)
        ref = np.asarray(((6, 3), (2, 7), (7, 2), (4, 5), (7, 2), (8, 1), (7, 2), (8, 1)))
        self.assertTrue(np.allclose(ref, output))

        output2 = tree.propagate_sequential(input_array, condition)
        ref2 = np.asarray(((8, 1), (2, 7), (7, 2), (4, 5), (7, 2), (8, 1), (7, 2), (8, 1)))
        self.assertTrue(np.allclose(ref2, output2))


if __name__ == '__main__':
    unittest.main()
