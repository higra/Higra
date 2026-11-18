############################################################################
# Copyright ESIEE Paris (2026)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import os
import tempfile
import unittest

import numpy as np

import higra as hg


def layouts(x):
    """
    Non-contiguous arrays holding the same values as ``x``: every second element of a larger array, reversed
    storage, a component of an array of pairs and, for arrays of dimension 2 or more, Fortran order and a
    transposed array.
    """
    x = np.asarray(x)
    result = []

    big = np.zeros((2 * x.shape[0],) + x.shape[1:], dtype=x.dtype)
    big[::2] = x
    result.append(("step2", big[::2]))

    result.append(("reversed", np.ascontiguousarray(x[::-1])[::-1]))

    result.append(("component", np.stack([x, x], axis=-1)[..., 0]))

    if x.ndim >= 2:
        result.append(("fortran", np.asfortranarray(x)))
        result.append(("transposed", np.ascontiguousarray(np.swapaxes(x, 0, 1)).swapaxes(0, 1)))

    for name, v in result:
        assert np.array_equal(v, x) and v.dtype == x.dtype, name
        assert x.size <= 1 or not v.flags.c_contiguous, name
    return result


class TestNonContiguousInputs(unittest.TestCase):
    """
    Non-contiguous arrays give the same results as contiguous arrays. Most functions below raised
    "NumPy: passing container with bad strides for layout" on non-contiguous arrays in earlier versions.
    """

    @staticmethod
    def tree():
        # 8 leaves, 6 internal nodes
        return hg.Tree((8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 13, 13))

    def assert_same(self, expected, actual, msg):
        expected = np.asarray(expected)
        actual = np.asarray(actual)
        self.assertEqual(expected.dtype, actual.dtype, msg)
        self.assertEqual(expected.shape, actual.shape, msg)
        self.assertTrue(np.array_equal(expected, actual), msg)

    def check(self, f, *args):
        """
        Calls ``f(*args)`` and ``f`` with each array argument replaced by each of its non-contiguous layouts.
        """
        expected = f(*args)
        tested = 0
        for i, a in enumerate(args):
            if isinstance(a, np.ndarray):
                for name, v in layouts(a):
                    new_args = list(args)
                    new_args[i] = v
                    self.assert_same(expected, f(*new_args), "argument %d, layout %s" % (i, name))
                    tested += 1
        self.assertGreater(tested, 0)
        return expected

    def test_tree_accumulators(self):
        t = self.tree()
        np.random.seed(1)
        scalar = np.random.randint(0, 10, t.num_vertices()).astype(np.float64)
        vector = np.random.randint(0, 10, (t.num_vertices(), 3)).astype(np.int32)
        leaf_scalar = scalar[:t.num_leaves()].copy()
        leaf_vector = vector[:t.num_leaves()].copy()
        condition = np.array([True, False] * (t.num_vertices() // 2))

        for w, lw in ((scalar, leaf_scalar), (vector, leaf_vector)):
            for acc in (hg.Accumulators.sum, hg.Accumulators.max, hg.Accumulators.mean):
                self.check(lambda a: hg.accumulate_parallel(t, a, acc), w)
                self.check(lambda a: hg.accumulate_sequential(t, a, acc), lw)
                for f in (hg.accumulate_and_add_sequential, hg.accumulate_and_multiply_sequential,
                          hg.accumulate_and_min_sequential, hg.accumulate_and_max_sequential):
                    self.check(lambda a, b: f(t, a, b, acc), w, lw)
                self.check(lambda a: hg.propagate_sequential_and_accumulate(t, a, acc), w)
                self.check(lambda a, c: hg.propagate_sequential_and_accumulate(t, a, acc, c), w, condition)

            self.check(lambda a: hg.propagate_parallel(t, a), w)
            self.check(lambda a, c: hg.propagate_parallel(t, a, c), w, condition)
            self.check(lambda a, c: hg.propagate_sequential(t, a, c), w, condition)

    def test_reconstruct_leaf_data(self):
        t = self.tree()
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 3, 4, 5), dtype=np.float64)
        self.check(lambda a: hg.reconstruct_leaf_data(t, a), altitudes)
        self.check(lambda a: hg.reconstruct_leaf_data(t, a), np.stack([altitudes, -altitudes], 1))

    def test_accumulate_at(self):
        indices = np.asarray((0, 1, 1, 2, -1, 0, 2, 2), dtype=np.int64)
        weights = np.arange(8, dtype=np.float32)
        weights2 = np.arange(16, dtype=np.int32).reshape(8, 2)
        for acc in (hg.Accumulators.sum, hg.Accumulators.max, hg.Accumulators.counter):
            self.check(lambda i, w: hg.accumulate_at(i, w, acc), indices, weights)
            self.check(lambda i, w: hg.accumulate_at(i, w, acc), indices, weights2)

    def test_graph_accumulators(self):
        g = hg.get_4_adjacency_graph((3, 4))
        edge_weights = np.arange(g.num_edges(), dtype=np.float64)
        vertex_weights = np.arange(2 * g.num_vertices(), dtype=np.uint8).reshape(-1, 2)
        for acc in (hg.Accumulators.sum, hg.Accumulators.min):
            self.check(lambda w: hg.accumulate_graph_edges(g, w, acc), edge_weights)
            self.check(lambda w: hg.accumulate_graph_vertices(g, w, acc), vertex_weights)

    def test_accumulate_on_contours(self):
        g = hg.get_4_adjacency_graph((2, 4))
        t = self.tree()
        node_weights = np.arange(t.num_vertices(), dtype=np.float64)
        self.check(lambda w: hg.accumulate_on_contours(t, w, hg.Accumulators.sum, g), node_weights)
        self.check(lambda w: hg.accumulate_on_contours(t, w, hg.Accumulators.max, g),
                   np.stack([node_weights, 2 * node_weights], 1))

    def test_weight_graph(self):
        shape = (4, 5)
        g = hg.get_4_adjacency_graph(shape)
        np.random.seed(2)
        image = np.random.rand(*shape, 3)
        for wf in (hg.WeightFunction.L2, hg.WeightFunction.L1, hg.WeightFunction.L_infinity):
            self.check(lambda v: hg.weight_graph(g, v, wf), image.reshape(-1, 3))
        for wf in (hg.WeightFunction.L2, hg.WeightFunction.max, hg.WeightFunction.mean, hg.WeightFunction.source):
            self.check(lambda v: hg.weight_graph(g, v, wf), image[:, :, 0].reshape(-1))

        # image channel and transposed image, as passed by users
        expected = hg.weight_graph(g, np.ascontiguousarray(image[:, :, 1]), hg.WeightFunction.L1)
        self.assert_same(expected, hg.weight_graph(g, image[:, :, 1], hg.WeightFunction.L1), "channel")
        gt = hg.get_4_adjacency_graph(shape[::-1])
        expected = hg.weight_graph(gt, np.ascontiguousarray(image[:, :, 1].T), hg.WeightFunction.L1)
        self.assert_same(expected, hg.weight_graph(gt, image[:, :, 1].T, hg.WeightFunction.L1), "transposed")

    def test_rag(self):
        g = hg.get_4_adjacency_graph((3, 4))
        labels = np.asarray((1, 1, 5, 5,
                             1, 1, 5, 5,
                             2, 2, 2, 7), dtype=np.int32)

        def rag_lab(lab):
            r = hg.make_region_adjacency_graph_from_labelisation(g, lab)
            return np.concatenate((hg.CptRegionAdjacencyGraph.get_vertex_map(r),
                                   hg.CptRegionAdjacencyGraph.get_edge_map(r)))

        self.check(rag_lab, labels)

        cut = hg.labelisation_2_graph_cut(g, labels).astype(np.float64)

        def rag_cut(c):
            r = hg.make_region_adjacency_graph_from_graph_cut(g, c)
            return np.concatenate((hg.CptRegionAdjacencyGraph.get_vertex_map(r),
                                   hg.CptRegionAdjacencyGraph.get_edge_map(r)))

        self.check(rag_cut, cut)

        rag = hg.make_region_adjacency_graph_from_labelisation(g, labels)
        rag_vertex_weights = np.arange(rag.num_vertices(), dtype=np.float32)
        rag_edge_weights = np.arange(2 * rag.num_edges(), dtype=np.int64).reshape(-1, 2)
        self.check(lambda w: hg.rag_back_project_vertex_weights(rag, w), rag_vertex_weights)
        self.check(lambda w: hg.rag_back_project_edge_weights(rag, w), rag_edge_weights)
        self.check(lambda w: hg.rag_accumulate_on_vertices(rag, hg.Accumulators.sum, w),
                   np.arange(g.num_vertices(), dtype=np.float64))

    def test_other_functions(self):
        """Representative functions whose bindings already accepted non-contiguous arrays."""
        shape = (5, 6)
        g = hg.get_4_adjacency_graph(shape)
        np.random.seed(3)
        image = np.random.randint(0, 20, shape).astype(np.float64)
        edge_weights = hg.weight_graph(g, image.reshape(-1), hg.WeightFunction.L1)

        def bpt(w):
            t, alt = hg.bpt_canonical(g, w)
            return np.concatenate((t.parents(), alt))

        self.check(bpt, edge_weights)

        def max_tree(im):
            t, alt = hg.component_tree_max_tree(g, im)
            return np.concatenate((t.parents(), alt))

        self.check(max_tree, image.reshape(-1))
        self.check(max_tree, image)

        def tos(im):
            t, alt = hg.component_tree_tree_of_shapes_image2d(im)
            return np.concatenate((t.parents(), alt))

        self.check(tos, image)

        def watershed(w):
            t, alt = hg.watershed_hierarchy_by_area(g, w)
            return np.concatenate((t.parents(), alt))

        self.check(watershed, edge_weights)

        t, altitudes = hg.bpt_canonical(g, edge_weights)
        self.check(lambda a: hg.attribute_volume(t, a), altitudes)
        self.check(lambda a: hg.labelisation_horizontal_cut_from_threshold(t, a, 3), altitudes)
        self.check(lambda a: hg.HorizontalCutExplorer(t, a).horizontal_cut_from_num_regions(4).nodes(), altitudes)

        def from_parents(p):
            return hg.Tree(p).parents()

        self.check(from_parents, t.parents().copy())

        vertices = np.asarray((0, 3, 7, 12, 29), dtype=np.int64)
        self.check(lambda v: t.parent(v), vertices)
        self.check(lambda v: g.degree(v), vertices)
        self.check(lambda v: t.is_leaf(v), vertices)
        self.check(lambda v1, v2: t.lowest_common_ancestor(v1, v2), vertices, vertices[::-1].copy())
        lca = t.lowest_common_ancestor_preprocess()
        self.check(lambda v1, v2: lca.lca(v1, v2), vertices, vertices[::-1].copy())

        g2 = hg.UndirectedGraph(10)
        sources = np.asarray((0, 1, 2, 3), dtype=np.int64)
        targets = np.asarray((5, 6, 7, 8), dtype=np.int64)

        def add_edges(s, d):
            graph = hg.UndirectedGraph(10)
            graph.add_edges(s, d)
            return np.concatenate(graph.edge_list())

        self.check(add_edges, sources, targets)

    def test_save_tree_attributes(self):
        t = self.tree()
        attribute = np.arange(t.num_vertices(), dtype=np.float64)
        with tempfile.TemporaryDirectory() as d:
            for name, v in layouts(attribute):
                filename = os.path.join(d, name + ".tree")
                hg.save_tree(filename, t, {"a": v})
                t2, attributes = hg.read_tree(filename)
                self.assertTrue(np.array_equal(t2.parents(), t.parents()))
                self.assertTrue(np.array_equal(attributes["a"], attribute), name)


class TestDtypeDispatch(unittest.TestCase):
    """
    The dtype of a non-contiguous input selects the same C++ overload as a contiguous input of the same dtype:
    the result has the same dtype and the values are not converted.
    """

    def check_dtype(self, f, x):
        expected = np.asarray(f(x))
        for name, v in layouts(x):
            r = np.asarray(f(v))
            self.assertEqual(r.dtype, expected.dtype, "%s %s" % (x.dtype, name))
            self.assertTrue(np.array_equal(r, expected), "%s %s" % (x.dtype, name))
        return expected

    def test_dtype_dispatch(self):
        t = TestNonContiguousInputs.tree()
        g = hg.get_4_adjacency_graph((3, 4))
        # values that an int8 conversion would change
        base = np.asarray((300.75, 2.5, 1000, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
        for dtype in (np.float32, np.int32, np.uint8, np.float64, np.int16, np.uint64):
            x = base.astype(dtype)
            r = self.check_dtype(lambda a: hg.propagate_parallel(t, a), x)
            self.assertEqual(r.dtype, dtype)
            r = self.check_dtype(lambda a: hg.accumulate_parallel(t, a, hg.Accumulators.max), x)
            self.assertEqual(r.dtype, dtype)
            # weight_graph returns float64; the values show that the input was not converted
            self.check_dtype(lambda a: hg.weight_graph(g, a, hg.WeightFunction.source), x[:12])
            r = self.check_dtype(lambda a: hg.component_tree_max_tree(g, a)[1], x[:12])
            self.assertEqual(r.dtype, dtype)
            r = self.check_dtype(lambda a: hg.accumulate_at(np.arange(14) % 3, a, hg.Accumulators.max), x)
            self.assertEqual(r.dtype, dtype)

    def test_byte_swapped(self):
        t = TestNonContiguousInputs.tree()
        x = np.arange(t.num_vertices(), dtype=np.float32) + 0.5
        swapped = x.astype(x.dtype.newbyteorder())
        r = hg.propagate_parallel(t, swapped[::-1][::-1])
        self.assertEqual(r.dtype, np.float32)
        self.assertTrue(np.array_equal(r, hg.propagate_parallel(t, x)))


class TestInputErrors(unittest.TestCase):

    def test_rank_mismatch_raises_type_error(self):
        # these arguments must be 1d arrays: other ranks are rejected before any computation
        with self.assertRaises(TypeError):
            hg.cpp._bpt_canonical(np.zeros((2, 2), np.int64), np.zeros(4, np.int64), np.zeros(4, np.int64), 4)
        with self.assertRaises(TypeError):
            hg.cpp._is_bipartite_graph(np.zeros(3, np.int64), np.zeros((3, 1), np.int64), 4)
        with self.assertRaises(TypeError):
            hg.cpp._get_bipartite_matching_graph_contour_image_2d(np.zeros(4, np.bool_), np.zeros(4, np.bool_), 1)

    def test_empty_index_arrays(self):
        empty = np.zeros(0, np.int64)
        self.assertTrue(hg.cpp._is_bipartite_graph(empty, empty, 3))
        t = TestNonContiguousInputs.tree()
        lca = t.lowest_common_ancestor_preprocess()
        self.assertEqual(lca.lca(empty, empty).size, 0)

    def test_negative_index(self):
        t = TestNonContiguousInputs.tree()
        lca = t.lowest_common_ancestor_preprocess()
        with self.assertRaises(RuntimeError):
            lca.lca(np.asarray((0, -1))[::-1], np.asarray((0, 1)))


class TestCallbacks(unittest.TestCase):

    def test_watershed_hierarchy_by_attribute(self):
        g = hg.get_4_adjacency_graph((3, 4))
        np.random.seed(4)
        edge_weights = np.random.randint(0, 5, g.num_edges()).astype(np.float64)
        calls = []

        def area(tree, altitudes):
            calls.append(1)
            return hg.attribute_area(tree)

        t, alt = hg.watershed_hierarchy_by_attribute(g, edge_weights, area)
        self.assertEqual(len(calls), 1)

        # results that are non-contiguous, of another dtype or lists are accepted
        def area_float32_reversed(tree, altitudes):
            calls.append(1)
            return np.ascontiguousarray(hg.attribute_area(tree)[::-1].astype(np.float32))[::-1]

        def area_list(tree, altitudes):
            calls.append(1)
            return hg.attribute_area(tree).tolist()

        for f in (area_float32_reversed, area_list):
            calls.clear()
            t2, alt2 = hg.watershed_hierarchy_by_attribute(g, edge_weights, f)
            self.assertEqual(len(calls), 1)
            self.assertTrue(np.array_equal(t2.parents(), t.parents()))
            self.assertTrue(np.array_equal(alt2, alt))
        self.check_hg_area(g, edge_weights, t, alt)

    def check_hg_area(self, g, edge_weights, t, alt):
        t3, alt3 = hg.watershed_hierarchy_by_area(g, edge_weights)
        self.assertTrue(np.array_equal(t3.parents(), t.parents()))
        self.assertTrue(np.array_equal(alt3, alt))


if __name__ == '__main__':
    unittest.main()
