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

        input_array = np.asarray((0, 1, 0, 1, 2, 0, 1, 1))
        res = hg.accumulate_parallel(tree, input_array, hg.Accumulators.argmin)
        ref = np.asarray((-1, -1, -1, -1, -1, 0, 0, 0))
        self.assertTrue(np.allclose(ref, res))

        res = hg.accumulate_parallel(tree, input_array, hg.Accumulators.argmax)
        ref = np.asarray((-1, -1, -1, -1, -1, 1, 2, 1))
        self.assertTrue(np.allclose(ref, res))

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

        output = hg.propagate_parallel(tree, input_array)
        ref = np.asarray(((6, 3), (6, 3), (7, 2), (7, 2), (7, 2), (8, 1), (8, 1), (8, 1)))
        self.assertTrue(np.allclose(ref, output))

        output = hg.propagate_parallel(tree, input_array, condition=condition)
        ref = np.asarray(((6, 3), (2, 7), (7, 2), (4, 5), (7, 2), (8, 1), (7, 2), (8, 1)))
        self.assertTrue(np.allclose(ref, output))

        output2 = hg.propagate_sequential(tree, input_array, condition)
        ref2 = np.asarray(((8, 1), (2, 7), (7, 2), (4, 5), (7, 2), (8, 1), (7, 2), (8, 1)))
        self.assertTrue(np.allclose(ref2, output2))

        output2 = hg.propagate_sequential_and_accumulate(tree, input_array, hg.Accumulators.sum)
        ref2 = np.asarray(((15, 12), (16, 11), (18, 9), (19, 8), (20, 7), (14, 4), (15, 3), (8, 1)))
        self.assertTrue(np.allclose(ref2, output2))

        output2 = hg.propagate_sequential_and_accumulate(tree, input_array, hg.Accumulators.prod, condition)
        ref2 = np.asarray(((48, 24), (2, 7), (21, 12), (4, 5), (35, 8), (48, 3), (7, 2), (8,  1)))
        self.assertTrue(np.allclose(ref2, output2))


    def test_empty_tree(self):
        reference_tree = TestTreeAccumulators.get_tree()
        acc = hg.Accumulators.sum

        def calls(tree, data, condition):
            num_leaves = tree.num_leaves()
            leaf_data = data[:num_leaves]
            return {
                "accumulate_parallel": lambda: hg.accumulate_parallel(tree, data, acc),
                "accumulate_sequential": lambda: hg.accumulate_sequential(tree, leaf_data, acc),
                "accumulate_and_add_sequential":
                    lambda: hg.accumulate_and_add_sequential(tree, data, leaf_data, acc),
                "accumulate_and_multiply_sequential":
                    lambda: hg.accumulate_and_multiply_sequential(tree, data, leaf_data, acc),
                "accumulate_and_min_sequential":
                    lambda: hg.accumulate_and_min_sequential(tree, data, leaf_data, acc),
                "accumulate_and_max_sequential":
                    lambda: hg.accumulate_and_max_sequential(tree, data, leaf_data, acc),
                "propagate_parallel": lambda: hg.propagate_parallel(tree, data),
                "propagate_parallel condition": lambda: hg.propagate_parallel(tree, data, condition),
                "propagate_sequential": lambda: hg.propagate_sequential(tree, data, condition),
                "propagate_sequential_and_accumulate":
                    lambda: hg.propagate_sequential_and_accumulate(tree, data, acc),
                "propagate_sequential_and_accumulate condition":
                    lambda: hg.propagate_sequential_and_accumulate(tree, data, acc, condition),
            }

        for compute_children in (False, True):
            for dtype in (np.float64, np.int32):
                for trailing_shape in ((), (3,)):
                    tree = hg.Tree(np.array([], dtype=np.int64))
                    if compute_children:
                        self.assertEqual(tree.num_children().shape, (0,))  # computes the children
                    data = np.zeros((0,) + trailing_shape, dtype=dtype)
                    condition = np.zeros((0,), dtype=np.bool_)

                    reference_data = np.ones((reference_tree.num_vertices(),) + trailing_shape, dtype=dtype)
                    reference_condition = np.zeros((reference_tree.num_vertices(),), dtype=np.bool_)
                    references = calls(reference_tree, reference_data, reference_condition)

                    for name, call in calls(tree, data, condition).items():
                        with self.subTest(function=name, dtype=dtype, shape=data.shape,
                                          compute_children=compute_children):
                            result = call()
                            self.assertEqual(result.shape, data.shape)
                            self.assertEqual(result.dtype, references[name]().dtype)


    def test_tree_accumulators_single_node_tree(self):
        for compute_children in (False, True):
            tree = hg.Tree(np.asarray((0,)))
            if compute_children:
                tree.children(0)
            res = hg.accumulate_parallel(tree, np.asarray((1.,)), hg.Accumulators.mean)
            self.assertTrue(res.shape == (1,))
            res = hg.accumulate_parallel(tree, np.ones((1, 3)), hg.Accumulators.sum)
            self.assertTrue(res.shape == (1, 3))
            res = hg.accumulate_sequential(tree, np.asarray((2.,)), hg.Accumulators.max)
            np.testing.assert_array_equal(res, (2.,))
            # the root is a leaf: it takes the leaf value
            res = hg.accumulate_and_add_sequential(tree, np.asarray((1.,)), np.asarray((2.,)), hg.Accumulators.sum)
            np.testing.assert_array_equal(res, (2.,))
            res = hg.accumulate_and_multiply_sequential(tree, np.asarray((3.,)), np.asarray((2.,)),
                                                        hg.Accumulators.sum)
            np.testing.assert_array_equal(res, (2.,))
            res = hg.accumulate_and_min_sequential(tree, np.asarray((1.,)), np.asarray((2.,)), hg.Accumulators.sum)
            np.testing.assert_array_equal(res, (2.,))
            res = hg.accumulate_and_max_sequential(tree, np.asarray((3.,)), np.asarray((2.,)), hg.Accumulators.sum)
            np.testing.assert_array_equal(res, (2.,))

    def test_accumulate_parallel_leaves(self):
        tree = TestTreeAccumulators.get_tree()
        for compute_children in (False, True):
            if compute_children:
                tree.children(0)
            for data in (np.asarray((3, 1, 2, 0, 5, 1, 1, 1), dtype=np.float64),
                         np.arange(16, dtype=np.int32).reshape(8, 2)):
                for acc in (hg.Accumulators.first, hg.Accumulators.last, hg.Accumulators.sum, hg.Accumulators.mean):
                    res = hg.accumulate_parallel(tree, data, acc)
                    self.assertEqual(res.dtype, data.dtype)
                    np.testing.assert_array_equal(res[:5], np.zeros_like(data[:5]))

    def test_accumulate_parallel_mean(self):
        tree = TestTreeAccumulators.get_tree()
        data = np.arange(8, dtype=np.float64)
        ref = (0, 0, 0, 0, 0, 0.5, 3, 5.5)
        np.testing.assert_allclose(hg.accumulate_parallel(tree, data, hg.Accumulators.mean), ref)
        tree.children(0)
        np.testing.assert_allclose(hg.accumulate_parallel(tree, data, hg.Accumulators.mean), ref)
        np.testing.assert_allclose(hg.accumulate_sequential(tree, data[:5], hg.Accumulators.mean),
                                   (0, 1, 2, 3, 4, 0.5, 3, 1.75))

    def test_index_accumulators(self):
        tree = TestTreeAccumulators.get_tree()
        scalar = np.asarray((3, 1, 2, 0, 5, 1, 1, 1), dtype=np.uint8)
        vectorial = np.asarray(((1, 1), (0, 0), (2, 2), (0, 1), (5, 5), (1, 1), (1, 1), (0, 0)), dtype=np.float32)
        for compute_children in (False, True):
            if compute_children:
                tree.children(0)
            for data, ref_min, ref_max in ((scalar, (1, 1, 0), (0, 2, 0)), (vectorial, (1, 1, 1), (0, 2, 1))):
                res = hg.accumulate_parallel(tree, data, hg.Accumulators.argmin)
                self.assertEqual(res.dtype, np.int64)
                np.testing.assert_array_equal(res, (-1,) * 5 + ref_min)
                res = hg.accumulate_parallel(tree, data, hg.Accumulators.argmax)
                self.assertEqual(res.dtype, np.int64)
                np.testing.assert_array_equal(res, (-1,) * 5 + ref_max)
                res = hg.accumulate_parallel(tree, data, hg.Accumulators.counter)
                self.assertEqual(res.dtype, np.int64)
                np.testing.assert_array_equal(res, (0, 0, 0, 0, 0, 2, 3, 2))

        # indices and counts do not depend on the range of the input type
        tree = hg.Tree(np.full(301, 300))
        data = np.ones(301, dtype=np.uint8)
        data[299] = 0
        self.assertEqual(hg.accumulate_parallel(tree, data, hg.Accumulators.argmin)[300], 299)
        self.assertEqual(hg.accumulate_parallel(tree, data, hg.Accumulators.counter)[300], 300)

    def test_sequential_functions_reject_index_accumulators(self):
        tree = TestTreeAccumulators.get_tree()
        data = np.arange(8, dtype=np.float64)
        leaf_data = np.arange(5, dtype=np.float64)
        condition = np.asarray((True, False, True, False, True, True, False, False))
        for acc in (hg.Accumulators.argmin, hg.Accumulators.argmax):
            with self.assertRaises(RuntimeError):
                hg.accumulate_sequential(tree, leaf_data, acc)
            with self.assertRaises(RuntimeError):
                hg.accumulate_and_add_sequential(tree, data, leaf_data, acc)
            with self.assertRaises(RuntimeError):
                hg.accumulate_and_max_sequential(tree, data, leaf_data, acc)
            with self.assertRaises(RuntimeError):
                hg.propagate_sequential_and_accumulate(tree, data, acc)
            with self.assertRaises(RuntimeError):
                hg.propagate_sequential_and_accumulate(tree, data, acc, condition)

    def test_counter_in_sequential_functions(self):
        tree = TestTreeAccumulators.get_tree()
        data = np.asarray((1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5))
        condition = np.asarray((True, False, True, False, True, True, False, False))
        res = hg.propagate_sequential_and_accumulate(tree, data, hg.Accumulators.counter, condition)
        self.assertEqual(res.dtype, np.float64)
        np.testing.assert_array_equal(res, (2, 2.5, 2, 4.5, 2, 2, 7.5, 8.5))


if __name__ == '__main__':
    unittest.main()
