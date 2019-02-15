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
import numpy as np
import higra as hg


class TestTreeAccumulators(unittest.TestCase):

    @staticmethod
    def get_tree():
        parent_relation = np.asarray((5, 5, 6, 6, 6, 7, 7, 7), dtype=np.uint64)
        return hg.Tree(parent_relation)

    def test_tree_accumulator(self):
        tree = TestTreeAccumulators.get_tree()
        input_array = np.asarray((1, 1, 1, 1, 1, 1, 1, 1))
        hg.CptValuedHierarchy.link(input_array, tree)

        res1 = hg.accumulate_parallel(input_array, hg.Accumulators.sum)
        ref1 = np.asarray((0, 0, 0, 0, 0, 2, 3, 2))
        self.assertTrue(np.allclose(ref1, res1))

        leaf_data = np.asarray((1, 1, 1, 1, 1))
        res2 = hg.accumulate_sequential(leaf_data, hg.Accumulators.sum, tree)
        ref2 = np.asarray((1, 1, 1, 1, 1, 2, 3, 5))
        self.assertTrue(np.allclose(ref2, res2))

        res3 = hg.accumulate_and_add_sequential(input_array, leaf_data, hg.Accumulators.max)
        ref3 = np.asarray((1, 1, 1, 1, 1, 2, 2, 3))
        self.assertTrue(np.allclose(ref3, res3))

        input_array = np.asarray((1, 2, 1, 2, 1, 1, 4, 5))
        hg.CptValuedHierarchy.link(input_array, tree)
        res4 = hg.accumulate_and_max_sequential(input_array, leaf_data, hg.Accumulators.sum)
        ref4 = np.asarray((1, 1, 1, 1, 1, 2, 4, 6))
        self.assertTrue(np.allclose(ref4, res4))

        input_array = np.asarray((1, 2, 1, 2, 1, 2, 3, 1))
        hg.CptValuedHierarchy.link(input_array, tree)
        res5 = hg.accumulate_and_multiply_sequential(input_array, leaf_data, hg.Accumulators.sum)
        ref5 = np.asarray((1, 1, 1, 1, 1, 4, 9, 13))
        self.assertTrue(np.allclose(ref5, res5))

        input_array = np.asarray((1, 2, 1, 2, 1, 4, 2, 10))
        hg.CptValuedHierarchy.link(input_array, tree)
        res6 = hg.accumulate_and_min_sequential(input_array, leaf_data, hg.Accumulators.sum)
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
        hg.CptValuedHierarchy.link(input_array, tree)

        res1 = hg.accumulate_parallel(input_array, hg.Accumulators.sum)
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
        res2 = hg.accumulate_sequential(leaf_data, hg.Accumulators.sum, tree)
        ref2 = np.asarray(((1, 0),
                           (1, 1),
                           (1, 2),
                           (1, 3),
                           (1, 4),
                           (2, 1),
                           (3, 9),
                           (5, 10)))
        self.assertTrue(np.allclose(ref2, res2))

        res3 = hg.accumulate_and_add_sequential(input_array, leaf_data, hg.Accumulators.sum)
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
        hg.CptValuedHierarchy.link(input_array, tree)
        condition = np.asarray((True, False, True, False, True, True, False, False))

        output = hg.propagate_parallel(input_array)
        ref = np.asarray(((6, 3), (6, 3), (7, 2), (7, 2), (7, 2), (8, 1), (8, 1), (8, 1)))
        self.assertTrue(np.allclose(ref, output))

        output = hg.propagate_parallel(input_array, condition=condition)
        ref = np.asarray(((6, 3), (2, 7), (7, 2), (4, 5), (7, 2), (8, 1), (7, 2), (8, 1)))
        self.assertTrue(np.allclose(ref, output))

        output2 = hg.propagate_sequential(input_array, condition)
        ref2 = np.asarray(((8, 1), (2, 7), (7, 2), (4, 5), (7, 2), (8, 1), (7, 2), (8, 1)))
        self.assertTrue(np.allclose(ref2, output2))

        output2 = hg.propagate_sequential_and_accumulate(input_array, hg.Accumulators.sum)
        ref2 = np.asarray(((15, 12), (16, 11), (18, 9), (19, 8), (20, 7), (14, 4), (15, 3), (8, 1)))
        self.assertTrue(np.allclose(ref2, output2))


if __name__ == '__main__':
    unittest.main()
