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
        self.graph = hg.get_4_adjacency_graph((2, 3))
        self.weights = np.array([1, 0, 2, 1, 1, 1, 2])

    def assert_exact(self, graph, weights=None, order=None):
        expected, levels = hg.bpt_canonical(graph, weights, order)
        actual, altitudes = hg.bpt_canonical_boruvka(graph, weights, order)
        np.testing.assert_array_equal(actual.parents(), expected.parents())
        np.testing.assert_array_equal(actual.mst_edge_map, expected.mst_edge_map)
        np.testing.assert_array_equal(altitudes, levels)
        self.assertEqual(altitudes.dtype, levels.dtype)
        return actual, altitudes

    def assert_relaxed(self, graph, weights, order=None):
        exact, levels = hg.bpt_canonical_boruvka(graph, weights, order)
        tree, altitudes = hg.bpt_canonical_boruvka(graph, weights, order, stable=False)
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
                                result = hg.bpt_canonical_boruvka(
                                    graph, weights, order, return_altitudes, compute_mst, stable)
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
        tree, altitudes = hg.bpt_canonical_boruvka(self.graph, np.empty((7, 0, 2)))
        expected, _ = hg.bpt_canonical(self.graph, sorted_edge_indices=np.arange(7))
        np.testing.assert_array_equal(tree.parents(), expected.parents())
        self.assertEqual(altitudes.shape, (11, 0, 2))
        singleton, levels = hg.bpt_canonical_boruvka(([], [], 1), sorted_edge_indices=[])
        np.testing.assert_array_equal(singleton.parents(), [0])
        np.testing.assert_array_equal(levels, [0])

    def test_graph_result_lifetime(self):
        for stable in (False, True):
            graph = hg.get_4_adjacency_graph((2, 3))
            weights = self.weights.copy()
            tree, altitudes = hg.bpt_canonical_boruvka(graph, weights, compute_mst=False, stable=stable)
            expected = hg.saliency(tree, altitudes).copy()
            weights[:] = 99
            del graph, weights
            gc.collect()
            mst = hg.CptBinaryHierarchy.get_mst(tree)
            self.assertEqual(mst.num_edges(), 5)
            np.testing.assert_array_equal(hg.saliency(tree, altitudes), expected)

    def test_input_mst_preserves_existing_metadata_behavior(self):
        original, _ = hg.bpt_canonical(self.graph, self.weights)
        mst = original.mst
        weights = self.weights[original.mst_edge_map]
        for compute in (False, True):
            old, _ = hg.bpt_canonical(mst, weights, compute_mst=compute)
            new, _ = hg.bpt_canonical_boruvka(mst, weights, compute_mst=compute)
            self.assertIs(hg.CptHierarchy.get_leaf_graph(new), self.graph)
            np.testing.assert_array_equal(new.mst_edge_map, old.mst_edge_map)
            # Existing API retains input-local maps even when linking to the base graph.
            old_mst = hg.CptBinaryHierarchy.get_mst(old)
            new_mst = hg.CptBinaryHierarchy.get_mst(new)
            np.testing.assert_array_equal(new_mst.sources(), old_mst.sources())
            np.testing.assert_array_equal(new_mst.targets(), old_mst.targets())
            self.assertIs(new_mst.base_graph, self.graph)

    def test_no_full_edge_sort_or_kruskal(self):
        with patch.object(hg, 'arg_sort', side_effect=AssertionError("full sort")), \
                patch.object(hg.cpp, '_bpt_canonical', side_effect=AssertionError("Kruskal")):
            hg.bpt_canonical_boruvka(self.graph, self.weights)
            hg.bpt_canonical_boruvka(self.graph, self.weights, stable=False)

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
        image_tree, image_levels = hg.bpt_canonical_boruvka(self.graph, self.weights, stable=False)
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
                    hg.bpt_canonical_boruvka(self.graph, self.weights, order, stable=stable)
            for weights in (np.zeros(6), np.zeros(()), np.zeros(7, complex),
                            np.zeros(7, object), np.zeros(7, np.float16), np.full((7, 2, 2), np.nan)):
                with self.subTest(shape=weights.shape), self.assertRaises(ValueError):
                    hg.bpt_canonical_boruvka(self.graph, weights, stable=stable)
            for altitudes in (False, True):
                with self.assertRaises(ValueError):
                    hg.bpt_canonical_boruvka(self.graph, np.full(7, np.nan), np.arange(7),
                                               return_altitudes=altitudes, stable=stable)
            for graph in (([], [], 0), ([0], [1], 3), ([0], [2], 2),
                          ([-1], [0], 2), ([0, 1], [1], 2), ([[0]], [[1]], 2),
                          ([0.5], [1], 2), ([0], [1], 2.5), ([2**64-1], [0], 2)):
                with self.subTest(graph=graph), self.assertRaises((ValueError, RuntimeError)):
                    hg.bpt_canonical_boruvka(graph, sorted_edge_indices=np.arange(len(graph[0])), stable=stable)
        with self.assertRaises(ValueError):
            hg.bpt_canonical_boruvka(self.graph)
        with self.assertRaises(ValueError):
            hg.bpt_canonical_boruvka(None, self.weights)

    def test_singleton_and_ownership(self):
        for stable in (False, True):
            for graph, weights in ((([], [], 1), np.empty(0)), (([0], [0], 1), np.array([np.inf]))):
                tree, altitudes = hg.bpt_canonical_boruvka(graph, weights, stable=stable)
                np.testing.assert_array_equal(tree.parents(), [0])
                np.testing.assert_array_equal(altitudes, [0])
                self.assertEqual(tree.mst_edge_map.size, 0)
            with self.assertRaises(ValueError):
                hg.bpt_canonical_boruvka(([0], [0], 1), [np.nan], stable=stable)
            sources = np.arange(4)
            targets = np.arange(1, 5)
            weights = np.array([4, 3, 10, 1])
            tree, altitudes = hg.bpt_canonical_boruvka((sources, targets, 5), weights, stable=stable)
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
