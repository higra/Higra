import sys
import unittest

import numpy as np

sys.path.insert(0, "@PYTHON_MODULE_PATH@")

import higra as hg


class TestTreeAccumulators(unittest.TestCase):

    @staticmethod
    def get_tree():
        parent_relation = np.asarray((5, 5, 6, 6, 6, 7, 7, 7), dtype=np.uint64)
        return hg.Tree(parent_relation)

    def test_tree_accumulator(self):
        tree = TestTreeAccumulators.get_tree()
        input_array = np.asarray((1, 1, 1, 1, 1, 1, 1, 1))

        res1 = hg.accumulate_parallel(tree, input_array, hg.Accumulators.sum)
        ref1 = np.asarray((0, 0, 0, 0, 0, 2, 3, 2))
        self.assertTrue(np.allclose(ref1, res1))

        leaf_data = np.asarray((1, 1, 1, 1, 1))
        res2 = hg.accumulate_sequential(tree, leaf_data, hg.Accumulators.sum)
        ref2 = np.asarray((1, 1, 1, 1, 1, 2, 3, 5))
        self.assertTrue(np.allclose(ref2, res2))

        res3 = hg.accumulate_and_add_sequential(tree, input_array, leaf_data, hg.Accumulators.max)
        ref3 = np.asarray((1, 1, 1, 1, 1, 2, 2, 3))
        self.assertTrue(np.allclose(ref3, res3))

        input_array = np.asarray((1, 2, 1, 2, 1, 1, 4, 5))
        res4 = hg.accumulate_and_max_sequential(tree, input_array, leaf_data, hg.Accumulators.sum)
        ref4 = np.asarray((1, 1, 1, 1, 1, 2, 4, 6))
        self.assertTrue(np.allclose(ref4, res4))

        input_array = np.asarray((1, 2, 1, 2, 1, 2, 3, 1))
        res5 = hg.accumulate_and_multiply_sequential(tree, input_array, leaf_data, hg.Accumulators.sum)
        ref5 = np.asarray((1, 1, 1, 1, 1, 4, 9, 13))
        self.assertTrue(np.allclose(ref5, res5))

        input_array = np.asarray((1, 2, 1, 2, 1, 4, 2, 10))
        res6 = hg.accumulate_and_min_sequential(tree, input_array, leaf_data, hg.Accumulators.sum)
        ref6 = np.asarray((1, 1, 1, 1, 1, 2, 2, 4))
        self.assertTrue(np.allclose(ref6, res6))

    def test_tree_accumulatorVec(self):
        tree = TestTreeAccumulators.get_tree()
        input_array = np.asarray(((1, 0),
                                  (1, 1),
                                  (1, 2),
                                  (1, 3),
                                  (1, 4),
                                  (1, 5),
                                  (1, 6),
                                  (1, 7)))

        res1 = hg.accumulate_parallel(tree, input_array, hg.Accumulators.sum)
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
        res2 = hg.accumulate_sequential(tree, leaf_data, hg.Accumulators.sum)
        ref2 = np.asarray(((1, 0),
                           (1, 1),
                           (1, 2),
                           (1, 3),
                           (1, 4),
                           (2, 1),
                           (3, 9),
                           (5, 10)))
        self.assertTrue(np.allclose(ref2, res2))

        res3 = hg.accumulate_and_add_sequential(tree, input_array, leaf_data, hg.Accumulators.sum)
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
        tree = TestTreeAccumulators.get_tree()
        input_array = np.asarray(((1, 8), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (8, 1)), dtype=np.float64)
        condition = np.asarray((True, False, True, False, True, True, False, False))

        output = hg.propagate_parallel(tree, input_array, condition)
        ref = np.asarray(((6, 3), (2, 7), (7, 2), (4, 5), (7, 2), (8, 1), (7, 2), (8, 1)))
        self.assertTrue(np.allclose(ref, output))

        output2 = hg.propagate_sequential(tree, input_array, condition)
        ref2 = np.asarray(((8, 1), (2, 7), (7, 2), (4, 5), (7, 2), (8, 1), (7, 2), (8, 1)))
        self.assertTrue(np.allclose(ref2, output2))


if __name__ == '__main__':
    unittest.main()
