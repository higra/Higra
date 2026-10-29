############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import gc
import unittest
from unittest.mock import patch

import numpy as np
import higra as hg


class TestBptBoruvka(unittest.TestCase):
    def setUp(self):
        self.compact_edges = False
        self.graph = hg.get_4_adjacency_graph((2, 3))
        self.weights = np.array([1, 0, 2, 1, 1, 1, 2])

    def boruvka(self, *args, **kwargs):
        return hg.bpt_canonical(*args, method="boruvka", compact_edges=self.compact_edges, **kwargs)

    def assert_exact(self, graph, weights=None, order=None):
        expected, levels = hg.bpt_canonical(graph, weights, order)
        actual, altitudes = self.boruvka(graph, weights, order)
        np.testing.assert_array_equal(actual.parents(), expected.parents())
        np.testing.assert_array_equal(actual.mst_edge_map, expected.mst_edge_map)
        np.testing.assert_array_equal(altitudes, levels)
        self.assertEqual(altitudes.dtype, levels.dtype)
        return actual, altitudes

    def assert_relaxed(self, graph, weights, order=None):
        exact, levels = self.boruvka(graph, weights, order)
        tree, altitudes = self.boruvka(graph, weights, order, stable=False)
        n = tree.num_leaves()
        np.testing.assert_array_equal(np.sort(tree.mst_edge_map), np.sort(exact.mst_edge_map))
        inverse = {e: n+i for i, e in enumerate(exact.mst_edge_map)}
        mapping = np.r_[np.arange(n), [inverse[e] for e in tree.mst_edge_map]]
        np.testing.assert_array_equal(mapping[tree.parents()], exact.parents()[mapping])
        np.testing.assert_array_equal(altitudes, levels[mapping])
        self.assertEqual(tree.category(), hg.TreeCategory.PartitionTree)
        parents = tree.parents()
        self.assertTrue(np.all(parents[:-1] > np.arange(parents.size-1)))
        self.assertEqual(parents[-1], parents.size-1)
        for node in range(n, tree.num_vertices()):
            self.assertEqual(len(tree.children(node)), 2)
        sources, targets = graph.edge_list() if hasattr(graph, "edge_list") else graph[:2]
        regions = [{i} for i in range(n)]
        for i, e in enumerate(tree.mst_edge_map):
            a, b = tree.children(n+i)
            self.assertTrue((sources[e] in regions[a] and targets[e] in regions[b]) or
                            (sources[e] in regions[b] and targets[e] in regions[a]))
            regions.append(regions[a] | regions[b])
        return exact, levels, tree, altitudes

    def test_dtypes_shapes_and_strides(self):
        for dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32,
                      np.uint32, np.int64, np.uint64, np.float32, np.float64):
            for shape in ((7,), (7, 3), (7, 2, 2)):
                with self.subTest(dtype=dtype, shape=shape):
                    weights = (np.arange(np.prod(shape)).reshape(shape) % 3).astype(dtype)
                    weights = np.repeat(weights, 2, axis=0)[::2]
                    weights = weights[::-1]
                    before = weights.copy()
                    self.assert_exact(self.graph, weights)
                    self.assert_relaxed(self.graph, weights)
                    np.testing.assert_array_equal(weights, before)
        for dtype, values in ((np.uint64, [2**64-1, 2**64-2, 2**63, 2**63+1, 0, 1, 2]),
                              (np.int64, [2**63-1, 2**63-2, -2**63, -2**63+1, 0, 1, 2])):
            self.assert_exact(self.graph, np.array(values, dtype=dtype))
        self.assert_exact(self.graph, self.weights.astype('>i8'))

    def test_flags_orders_and_concepts(self):
        order = np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.uint64)
        for stable in (True, False):
            for as_tuple in (True, False):
                graph = (*self.graph.edge_list(), self.graph.num_vertices()) if as_tuple else self.graph
                for weights in (None, self.weights):
                    for return_altitudes in (True, False):
                        for compute_mst in (True, False):
                            with self.subTest(stable=stable, tuple=as_tuple, weights=weights is not None,
                                              altitudes=return_altitudes, mst=compute_mst):
                                result = self.boruvka(
                                    graph, weights, order, return_altitudes, compute_mst, stable=stable)
                                tree = result[0] if return_altitudes else result
                                self.assertTrue(hg.CptBinaryHierarchy.validate(tree))
                                if return_altitudes:
                                    keys = weights
                                    if keys is None:
                                        keys = np.empty(7, dtype=np.int64)
                                        keys[order] = np.arange(7)
                                    np.testing.assert_array_equal(result[1][6:], keys[tree.mst_edge_map])
                                if not as_tuple:
                                    self.assertIs(hg.CptHierarchy.get_leaf_graph(tree), self.graph)
                                    self.assertEqual(tree.mst is not None, compute_mst)
                                    mst = hg.CptBinaryHierarchy.get_mst(tree)
                                    self.assertIs(mst.base_graph, self.graph)
                                    np.testing.assert_array_equal(mst.mst_edge_map, tree.mst_edge_map)
                                    np.testing.assert_array_equal(mst.sources(), self.graph.sources()[tree.mst_edge_map])
                                else:
                                    self.assertIsNone(tree.mst)
        self.assert_exact(self.graph, self.weights, order)
        self.assert_exact(self.graph, None, order)
        self.assert_relaxed(self.graph, self.weights, order)
        self.assert_relaxed(self.graph, None, order)

    def test_integer_orders_and_empty_features(self):
        sources, targets = self.graph.edge_list()
        for dtype in (np.int8, np.uint8, np.int32, np.uint32, np.int64, np.uint64):
            order = np.repeat(np.arange(7, dtype=dtype), 2)[::2][::-1]
            self.assert_exact((sources.astype(dtype), targets.astype(dtype), 6), None, order)
        tree, altitudes = self.boruvka(self.graph, np.empty((7, 0, 2)))
        expected, _ = hg.bpt_canonical(self.graph, sorted_edge_indices=np.arange(7))
        np.testing.assert_array_equal(tree.parents(), expected.parents())
        self.assertEqual(altitudes.shape, (11, 0, 2))
        singleton, levels = self.boruvka(([], [], 1), sorted_edge_indices=[])
        np.testing.assert_array_equal(singleton.parents(), [0])
        np.testing.assert_array_equal(levels, [0])

    def test_graph_result_lifetime(self):
        for stable in (False, True):
            graph = hg.get_4_adjacency_graph((2, 3))
            weights = self.weights.copy()
            tree, altitudes = self.boruvka(graph, weights, compute_mst=False, stable=stable)
            expected = hg.saliency(tree, altitudes).copy()
            weights[:] = 99
            del graph, weights
            gc.collect()
            mst = hg.CptBinaryHierarchy.get_mst(tree)
            self.assertEqual(mst.num_edges(), 5)
            np.testing.assert_array_equal(hg.saliency(tree, altitudes), expected)

    def assert_input_mst_result(self, input_mst, weights=None, order=None):
        base = input_mst.base_graph
        input_map = input_mst.mst_edge_map.copy()
        expected, levels = hg.bpt_canonical(input_mst, weights, order)
        for stable in (False, True):
            for eager in (False, True):
                for return_altitudes in (False, True):
                    with self.subTest(stable=stable, eager=eager, return_altitudes=return_altitudes):
                        result = self.boruvka(
                            input_mst, weights, order, compute_mst=eager,
                            return_altitudes=return_altitudes, stable=stable)
                        if return_altitudes:
                            tree, altitudes = result
                        else:
                            tree = result
                        self.assertEqual(tree.mst is not None, eager)
                        self.assertIs(hg.CptHierarchy.get_leaf_graph(tree), base)
                        np.testing.assert_array_equal(np.sort(tree.mst_edge_map), np.sort(input_map))
                        if stable:
                            np.testing.assert_array_equal(tree.parents(), expected.parents())
                            np.testing.assert_array_equal(tree.mst_edge_map, expected.mst_edge_map)
                        mst = hg.CptBinaryHierarchy.get_mst(tree)
                        self.assertIs(hg.CptBinaryHierarchy.get_mst(tree), mst)
                        self.assertIs(mst.base_graph, base)
                        np.testing.assert_array_equal(mst.mst_edge_map, tree.mst_edge_map)
                        np.testing.assert_array_equal(mst.sources(), base.sources()[tree.mst_edge_map])
                        np.testing.assert_array_equal(mst.targets(), base.targets()[tree.mst_edge_map])
                        for i, (source, target) in enumerate(zip(*mst.edge_list())):
                            self.assertEqual(tree.lowest_common_ancestor(source, target), tree.num_leaves() + i)
                        if return_altitudes:
                            local = np.array([np.flatnonzero(input_map == edge)[0]
                                              for edge in tree.mst_edge_map], dtype=np.int64)
                            if weights is None:
                                values = np.empty_like(order)
                                values[order] = np.arange(order.size)
                            else:
                                values = weights
                            np.testing.assert_array_equal(altitudes[tree.num_leaves():], values[local])
                            self.assertEqual(altitudes.dtype, values.dtype)
                            if stable:
                                np.testing.assert_array_equal(altitudes, levels)

    def test_input_mst_maps(self):
        graph = hg.UndirectedGraph(3)
        graph.add_edges([0, 0, 1], [1, 2, 2])
        mst = hg.minimum_spanning_tree(graph, np.array([10, 1, 2]))
        np.testing.assert_array_equal(mst.mst_edge_map, [1, 2])
        self.assert_input_mst_result(mst, np.array([1, 2]))
        for eager in (False, True):
            tree, _ = self.boruvka(mst, np.array([1, 2]), compute_mst=eager)
            rebuilt, _ = self.boruvka(hg.CptBinaryHierarchy.get_mst(tree),
                                                np.array([1, 2]), compute_mst=eager)
            np.testing.assert_array_equal(rebuilt.mst_edge_map, [1, 2])
            np.testing.assert_array_equal(rebuilt.parents(), tree.parents())

    def test_reordered_input_mst_weights_and_ranks(self):
        graph = hg.UndirectedGraph(4)
        graph.add_edges([0, 0, 1, 2, 0], [1, 2, 2, 3, 3])
        mst = hg.subgraph(graph, np.array([3, 1, 2]))
        hg.CptMinimumSpanningTree.link(mst, graph, np.array([3, 1, 2]))
        for dtype in (np.int32, np.uint64, np.float32, np.float64):
            for feature_shape in ((), (2,), (2, 2)):
                weights = np.broadcast_to(np.array([7, 5, 1], dtype=dtype).reshape(
                    (3,) + (1,) * len(feature_shape)), (3,) + feature_shape).copy()
                self.assert_input_mst_result(mst, weights)
        order = np.array([2, 0, 1])
        self.assert_input_mst_result(mst, order=order)
        self.assert_input_mst_result(mst, np.array([7, 5, 1]), order)

    def test_nested_and_singleton_input_mst(self):
        graph = hg.UndirectedGraph(3)
        graph.add_edges([0, 0, 1], [1, 2, 2])
        first = hg.minimum_spanning_tree(graph, np.array([10, 2, 1]))
        second = hg.minimum_spanning_tree(first, np.array([2, 1]))
        self.assert_input_mst_result(second, np.array([1, 2]))
        np.testing.assert_array_equal(first.mst_edge_map, [2, 1])
        self.assertIs(first.base_graph, graph)
        singleton = hg.minimum_spanning_tree(hg.UndirectedGraph(1), np.array([], dtype=np.float64))
        self.assert_input_mst_result(singleton, np.array([], dtype=np.float64))

    def test_no_full_edge_sort_or_kruskal(self):
        with patch.object(hg, 'arg_sort', side_effect=AssertionError("full sort")), \
                patch.object(hg.cpp, '_bpt_canonical', side_effect=AssertionError("Kruskal")):
            self.boruvka(self.graph, self.weights)
            self.boruvka(self.graph, self.weights, stable=False)

    def test_relaxed_consumers_and_threshold_oracle(self):
        graph = hg.UndirectedGraph(5)
        graph.add_edges(np.arange(4), np.arange(1, 5))
        exact, levels, tree, altitudes = self.assert_relaxed(graph, np.array([4., 3., 10., 1.]))
        self.assertTrue(np.any(np.diff(altitudes[5:]) < 0))
        np.testing.assert_array_equal(hg.saliency(tree, altitudes), hg.saliency(exact, levels))
        lca = tree.lowest_common_ancestor(np.repeat(np.arange(5), 5), np.tile(np.arange(5), 5))
        expected_lca = exact.lowest_common_ancestor(np.repeat(np.arange(5), 5), np.tile(np.arange(5), 5))
        np.testing.assert_array_equal(altitudes[lca], levels[expected_lca])
        for threshold in (0, 1, 2, 3, 4, 9, 10, 11):
            labels = hg.labelisation_horizontal_cut_from_threshold(tree, altitudes, threshold)
            # Independent connectivity oracle on original edges.
            reach = np.eye(5, dtype=bool)
            for e, w in enumerate([4, 3, 10, 1]):
                if w <= threshold:
                    reach[e, e+1] = reach[e+1, e] = True
            for k in range(5):
                reach |= reach[:, k, None] & reach[None, k, :]
            np.testing.assert_array_equal(labels[:, None] == labels, reach)
        for count in range(1, 6):
            a = hg.labelisation_horizontal_cut_from_num_regions(tree, altitudes, count)
            b = hg.labelisation_horizontal_cut_from_num_regions(exact, levels, count)
            np.testing.assert_array_equal(a[:, None] == a, b[:, None] == b)
        deleted = altitudes < 4
        expected = hg.reconstruct_leaf_data(exact, levels, levels < 4)
        np.testing.assert_array_equal(hg.reconstruct_leaf_data(tree, altitudes, deleted), expected)
        image_tree, image_levels = self.boruvka(self.graph, self.weights, stable=False)
        self.assertEqual(hg.reconstruct_leaf_data(image_tree, image_levels).shape, (2, 3))

    def test_random_ties_loops_and_infinities(self):
        rng = np.random.default_rng(42)
        for _ in range(30):
            graph = hg.UndirectedGraph(9)
            graph.add_edges(np.r_[np.arange(8), rng.integers(0, 9, 20)],
                            np.r_[np.arange(1, 9), rng.integers(0, 9, 20)])
            weights = rng.integers(0, 4, 28).astype(float)
            self.assert_exact(graph, weights)
            self.assert_relaxed(graph, weights)
        self.assert_exact(self.graph, np.array([-np.inf, np.inf, 0, np.inf, -np.inf, -0., 0]))
        self.assert_relaxed(self.graph, np.full(7, np.inf))
        self.assert_exact(self.graph, np.zeros(7))
        sources, targets = self.graph.edge_list()
        self.assert_exact((sources[::-1], targets[::-1], 6), self.weights[::-1])

    def test_validation(self):
        bad_orders = ([0]*7, [-1, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 7],
                      [0, 1], np.zeros((7, 1), dtype=int), np.arange(7.) , [2**64-1]*7)
        for stable in (True, False):
            for order in bad_orders:
                with self.subTest(order=order, stable=stable), self.assertRaises((ValueError, RuntimeError)):
                    self.boruvka(self.graph, self.weights, order, stable=stable)
            for weights in (np.zeros(6), np.zeros(()), np.zeros(7, complex),
                            np.zeros(7, object), np.zeros(7, np.float16), np.full((7, 2, 2), np.nan)):
                with self.subTest(shape=weights.shape), self.assertRaises(ValueError):
                    self.boruvka(self.graph, weights, stable=stable)
            for altitudes in (False, True):
                with self.assertRaises(ValueError):
                    self.boruvka(self.graph, np.full(7, np.nan), np.arange(7),
                                               return_altitudes=altitudes, stable=stable)
            for graph in (([], [], 0), ([0], [1], 3), ([0], [2], 2),
                          ([-1], [0], 2), ([0, 1], [1], 2), ([[0]], [[1]], 2),
                          ([0.5], [1], 2), ([0], [1], 2.5), ([2**64-1], [0], 2)):
                with self.subTest(graph=graph), self.assertRaises((ValueError, RuntimeError)):
                    self.boruvka(graph, sorted_edge_indices=np.arange(len(graph[0])), stable=stable)
        with self.assertRaises(ValueError):
            self.boruvka(self.graph)
        with self.assertRaises(ValueError):
            self.boruvka(None, self.weights)

    def test_singleton_and_ownership(self):
        for stable in (False, True):
            for graph, weights in ((([], [], 1), np.empty(0)), (([0], [0], 1), np.array([np.inf]))):
                tree, altitudes = self.boruvka(graph, weights, stable=stable)
                np.testing.assert_array_equal(tree.parents(), [0])
                np.testing.assert_array_equal(altitudes, [0])
                self.assertEqual(tree.mst_edge_map.size, 0)
            with self.assertRaises(ValueError):
                self.boruvka(([0], [0], 1), [np.nan], stable=stable)
            sources = np.arange(4)
            targets = np.arange(1, 5)
            weights = np.array([4, 3, 10, 1])
            tree, altitudes = self.boruvka((sources, targets, 5), weights, stable=stable)
            parents = tree.parents()
            edge_map = tree.mst_edge_map
            expected_parents = parents.copy()
            expected_map = edge_map.copy()
            expected_altitudes = altitudes.copy()
            sources[:] = targets[:] = weights[:] = 0
            del sources, targets, weights, tree
            gc.collect()
            np.testing.assert_array_equal(parents, expected_parents)
            np.testing.assert_array_equal(edge_map, expected_map)
            np.testing.assert_array_equal(altitudes, expected_altitudes)


class TestBptBoruvkaCompact(TestBptBoruvka):
    def setUp(self):
        super().setUp()
        self.compact_edges = True


class TestBptCanonicalDispatch(unittest.TestCase):
    def setUp(self):
        self.graph = hg.get_4_adjacency_graph((2, 3))
        self.weights = np.array([1, 0, 2, 1, 1, 1, 2])

    def assert_result_equal(self, actual, expected, return_altitudes=True):
        if return_altitudes:
            np.testing.assert_array_equal(actual[1], expected[1])
            self.assertEqual(actual[1].dtype, expected[1].dtype)
            actual, expected = actual[0], expected[0]
        np.testing.assert_array_equal(actual.parents(), expected.parents())
        np.testing.assert_array_equal(actual.mst_edge_map, expected.mst_edge_map)
        self.assertEqual(actual.mst is None, expected.mst is None)
        self.assertTrue(hg.CptBinaryHierarchy.validate(actual))

    def test_exports_and_legacy_calls(self):
        import higra.hierarchy as hierarchy
        self.assertIs(hierarchy.bpt_canonical_kruskal, hg.bpt_canonical_kruskal)
        for graph in (self.graph, (*self.graph.edge_list(), 6)):
            for order in (None, np.arange(7)[::-1]):
                for weights in ((self.weights,) if order is None else (None, self.weights)):
                    for return_altitudes in (False, True):
                        for compute_mst in (False, True):
                            args = (graph, weights, order, return_altitudes, compute_mst)
                            expected = hg.bpt_canonical_kruskal(*args)
                            self.assert_result_equal(hg.bpt_canonical(*args), expected, return_altitudes)
                            self.assert_result_equal(hg.bpt_canonical(*args, method="kruskal"),
                                                     expected, return_altitudes)
        for count in range(2, 6):
            args = (self.graph, self.weights, None, True, True)[:count]
            self.assert_result_equal(hg.bpt_canonical(*args), hg.bpt_canonical_kruskal(*args))

    def test_boruvka_routing_and_traversal_equivalence(self):
        for order in (None, np.arange(7)[::-1]):
            for stable in (False, True):
                expected = hg.bpt_canonical_boruvka(self.graph, self.weights, order, True, True, stable)
                for compact in (False, True):
                    actual = hg.bpt_canonical_boruvka(self.graph, self.weights, order, True, True,
                                                     stable, compact_edges=compact)
                    self.assert_result_equal(actual, expected)
                    self.assert_result_equal(hg.bpt_canonical(
                        self.graph, self.weights, order, method="boruvka", stable=stable,
                        compact_edges=compact), actual)
        self.assert_result_equal(hg.bpt_canonical(self.graph, self.weights, method="boruvka"),
                                 hg.bpt_canonical_boruvka(self.graph, self.weights, compact_edges=False))

    def test_specializations_forward_once_and_propagate_typeerror(self):
        from higra.hierarchy import hierarchy_core

        for method in ("kruskal", "boruvka"):
            algorithm = getattr(hg, "bpt_canonical_" + method)
            options = {} if method == "kruskal" else {"stable": True, "compact_edges": False}
            sentinel = object()
            with patch.object(hierarchy_core, "bpt_canonical", return_value=sentinel) as dispatcher:
                self.assertIs(algorithm(self.graph, self.weights, None, False, False), sentinel)
                dispatcher.assert_called_once_with(self.graph, self.weights, None, False, False,
                                                   method=method, **options)
            error = TypeError("from dispatcher")
            with patch.object(hierarchy_core, "bpt_canonical", side_effect=error) as dispatcher:
                with self.assertRaises(TypeError) as caught:
                    algorithm(self.graph, self.weights)
                self.assertIs(caught.exception, error)
                dispatcher.assert_called_once()
        # Verify both native paths receive defaults and explicit options unchanged.
        for stable in (False, True):
            for compact in (False, True):
                for name, order in (("_bpt_canonical_boruvka", None),
                                    ("_bpt_canonical_boruvka_from_order", np.arange(7))):
                    with patch.object(hg.cpp, name, wraps=getattr(hg.cpp, name)) as native:
                        hg.bpt_canonical(self.graph, self.weights, order, method="boruvka",
                                         stable=stable, compact_edges=compact)
                        native.assert_called_once()
                        self.assertEqual(native.call_args.args[-2:], (stable, compact))
        with patch.object(hg.cpp, "_bpt_canonical_boruvka",
                          wraps=hg.cpp._bpt_canonical_boruvka) as native:
            hg.bpt_canonical_boruvka(self.graph, self.weights)
            self.assertEqual(native.call_args.args[-2:], (True, False))

    def test_common_input_preparation(self):
        sources, targets = self.graph.edge_list()
        order = np.arange(7, dtype=np.uint32)[::-1]
        for method in ("kruskal", "boruvka"):
            for weights in (self.weights.tolist(), self.weights.astype(">i8"),
                            np.repeat(self.weights, 2)[::2],
                            np.broadcast_to(self.weights[:, None, None], (7, 2, 3)),
                            np.empty((7, 0, 2))):
                for indices in (None, order.tolist(), order, order.astype(">u4")):
                    with self.subTest(method=method, shape=np.shape(weights), order=indices):
                        actual = hg.bpt_canonical((sources.tolist(), targets.tolist(), 6),
                                                  weights, indices, method=method)
                        expected = hg.bpt_canonical(self.graph, np.asarray(weights), indices,
                                                    compute_mst=False)
                        self.assert_result_equal(actual, expected)
                        self.assertEqual(actual[1].shape[1:], np.shape(weights)[1:])
                        self.assertEqual(actual[1].dtype, np.asarray(weights).dtype)
            tree, ranks = hg.bpt_canonical(self.graph, sorted_edge_indices=order, method=method)
            self.assertEqual(ranks.dtype, order.dtype)
            np.testing.assert_array_equal(ranks[6:], np.argsort(order)[tree.mst_edge_map])
            tree, levels = hg.bpt_canonical(([], [], 1), sorted_edge_indices=[], method=method)
            np.testing.assert_array_equal(tree.parents(), [0])
            np.testing.assert_array_equal(levels, [0])
            tree, levels = hg.bpt_canonical(([], [], 1), np.empty(0), method=method)
            np.testing.assert_array_equal(tree.parents(), [0])
            np.testing.assert_array_equal(levels, [0])
            for bad in ([0.] * 7, [-1] * 7, [[0]] * 7, [2**63] * 7):
                for graph in ((bad, targets, 6), (sources, bad, 6)):
                    with self.subTest(method=method, graph=graph), self.assertRaises(ValueError):
                        hg.bpt_canonical(graph, self.weights, method=method)
                with self.subTest(method=method, order=bad), self.assertRaises(ValueError):
                    hg.bpt_canonical(self.graph, self.weights, bad, method=method)
            for weights in ([1], [np.nan] * 7, np.zeros(7, bool), np.zeros(7, np.float16)):
                for indices in (None, order):
                    with self.subTest(method=method, weights=weights), self.assertRaises(ValueError):
                        hg.bpt_canonical(self.graph, weights, indices, return_altitudes=False,
                                         method=method)

    def test_selector_and_argument_errors(self):
        for method in ("unknown", "Kruskal", "", None, 1, [], {}, np.array(["kruskal"])):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "method.*kruskal.*boruvka"):
                hg.bpt_canonical(self.graph, self.weights, method=method)
        for kwargs in ({"stable": True}, {"compact_edges": False}, {"unknown": 1}):
            with self.assertRaises(TypeError):
                hg.bpt_canonical(self.graph, self.weights, **kwargs)
        with self.assertRaises(TypeError):
            hg.bpt_canonical(self.graph, self.weights, method="boruvka", unknown=1)
        for algorithm in (hg.bpt_canonical, hg.bpt_canonical_kruskal, hg.bpt_canonical_boruvka):
            with self.assertRaises(TypeError):
                algorithm(self.graph, self.weights, edge_weights=self.weights)
        with self.assertRaises(TypeError):
            hg.bpt_canonical(self.graph, self.weights, None, True, True, "boruvka")
        with self.assertRaises(TypeError):
            hg.bpt_canonical_boruvka(self.graph, self.weights, None, True, True, True, True)
