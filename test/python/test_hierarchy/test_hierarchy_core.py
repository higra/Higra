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
import numpy as np


class TestHierarchyCore(unittest.TestCase):

    @staticmethod
    def getTree():
        parent_relation = np.asarray((5, 5, 6, 6, 6, 7, 7, 7), dtype=np.uint64)
        return hg.Tree(parent_relation)

    def test_BPTTrivial(self):
        graph = hg.get_4_adjacency_graph((1, 3))

        edge_weights = np.asarray([2, 3])

        tree, altitudes = hg.bpt_canonical(graph, edge_weights)
        mst = hg.CptBinaryHierarchy.construct(tree)["mst"]

        self.assertTrue(tree.num_vertices() == 5)
        self.assertTrue(tree.num_edges() == 4)
        self.assertTrue(np.allclose(tree.parents(), (3, 3, 4, 4, 4)))
        self.assertTrue(np.allclose(altitudes, (0, 0, 0, 2, 3)))
        self.assertTrue(mst.num_vertices() == 3)
        self.assertTrue(mst.num_edges() == 2)

    def test_BPT(self):
        graph = hg.get_4_adjacency_graph((2, 3))

        edge_weights = np.asarray((1, 0, 2, 1, 1, 1, 2))

        tree, altitudes = hg.bpt_canonical(graph, edge_weights)
        mst = hg.CptBinaryHierarchy.construct(tree)["mst"]
        mst_edge_map = getattr(mst, "mst_edge_map", None)

        self.assertTrue(tree.num_vertices() == 11)
        self.assertTrue(tree.num_edges() == 10)
        self.assertTrue(np.allclose(tree.parents(), (6, 7, 9, 6, 8, 9, 7, 8, 10, 10, 10)))
        self.assertTrue(np.allclose(altitudes, (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2)))
        self.assertTrue(mst.num_vertices() == 6)
        self.assertTrue(mst.num_edges() == 5)

        ref = [(0, 3),
               (0, 1),
               (1, 4),
               (2, 5),
               (1, 2)]
        test = []
        for e in mst.edges():
            test.append((e[0], e[1]))
        self.assertTrue(ref == test)

        self.assertTrue(np.all(mst_edge_map == (1, 0, 3, 4, 2)))

    def test_BPT_from_edge_list(self):
        graph = hg.get_4_adjacency_graph((2, 3))
        sources, targets = graph.edge_list()
        edge_weights = np.asarray((1, 0, 2, 1, 1, 1, 2))

        tree, altitudes = hg.bpt_canonical((sources, targets, graph.num_vertices()), edge_weights)
        mst_edge_map = tree.mst_edge_map

        self.assertTrue(tree.num_vertices() == 11)
        self.assertTrue(tree.num_edges() == 10)
        self.assertTrue(np.allclose(tree.parents(), (6, 7, 9, 6, 8, 9, 7, 8, 10, 10, 10)))
        self.assertTrue(np.allclose(altitudes, (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2)))

        self.assertTrue(np.all(mst_edge_map == (1, 0, 3, 4, 2)))

    def test_bpt_canonical_options(self):
        graph = hg.get_4_adjacency_graph((2, 3))
        edge_weights = np.asarray((1, 0, 2, 1, 1, 1, 2))
        sorted_edge_indices = np.asarray((1, 0, 3, 4, 5, 2, 6))
        ref_parents = np.asarray((6, 7, 9, 6, 8, 9, 7, 8, 10, 10, 10))
        ref_mst_edge_map = np.asarray((1, 0, 3, 4, 2))
        ref_altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2))
        ref_altitudes_no_weights = np.asarray((0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 5))

        tree = hg.bpt_canonical(graph, edge_weights, return_altitudes=False, compute_mst=False)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(getattr(tree, "mst", None) is None)
        self.assertTrue(np.all(tree.mst_edge_map == ref_mst_edge_map))
        mst = hg.CptBinaryHierarchy.get_mst(tree)
        self.assertTrue(mst.num_vertices() == 6)
        self.assertTrue(mst.num_edges() == 5)
        self.assertTrue(np.all(mst.mst_edge_map == ref_mst_edge_map))

        tree, altitudes = hg.bpt_canonical(graph, edge_weights, return_altitudes=True, compute_mst=True)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(tree.mst is not None)
        self.assertTrue(tree.mst.num_vertices() == 6)
        self.assertTrue(tree.mst.num_edges() == 5)
        self.assertTrue(np.all(tree.mst.mst_edge_map == ref_mst_edge_map))
        self.assertTrue(np.all(tree.mst_edge_map == ref_mst_edge_map))
        self.assertTrue(np.all(altitudes == ref_altitudes))

        tree, altitudes = hg.bpt_canonical(graph, edge_weights, sorted_edge_indices)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(altitudes == ref_altitudes))

        tree, altitudes = hg.bpt_canonical(graph, sorted_edge_indices=sorted_edge_indices)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(altitudes == ref_altitudes_no_weights))

    def test_bpt_canonical_vectorial(self):
        graph = hg.get_4_adjacency_graph((2, 3))
        edge_weights = np.asarray(((1, 0, 2, 1, 1, 1, 2),
                                   (0, 2, 0, 0, 1, 1, 0),
                                   (0, 5, 5, 1, 0, 1, 1))).T
        sorted_edge_indices = np.asarray((1, 0, 3, 4, 5, 6, 2))
        ref_parents = np.asarray((6, 7, 9, 6, 8, 9, 7, 8, 10, 10, 10))
        ref_mst_edge_map = np.asarray((1, 0, 3, 4, 6))
        ref_altitudes = np.asarray(((0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2),
                                    (0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0),
                                    (0, 0, 0, 0, 0, 0, 5, 0, 1, 0, 1))).T
        ref_altitudes_no_weights = np.asarray((0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 5))

        tree, altitudes = hg.bpt_canonical(graph, edge_weights)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(altitudes == ref_altitudes))
        self.assertTrue(np.all(tree.mst_edge_map == ref_mst_edge_map))

        tree, altitudes = hg.bpt_canonical(graph, edge_weights, sorted_edge_indices)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(altitudes == ref_altitudes))
        self.assertTrue(np.all(tree.mst_edge_map == ref_mst_edge_map))

        tree, altitudes = hg.bpt_canonical(graph, sorted_edge_indices=sorted_edge_indices)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(altitudes == ref_altitudes_no_weights))
        self.assertTrue(np.all(tree.mst_edge_map == ref_mst_edge_map))

    def assert_bpt_mst(self, tree, base_graph, expected_map, expected_edges):
        self.assertIs(hg.CptHierarchy.get_leaf_graph(tree), base_graph)
        np.testing.assert_array_equal(hg.CptBinaryHierarchy.get_mst_edge_map(tree), expected_map)
        mst = hg.CptBinaryHierarchy.get_mst(tree)
        self.assertIs(hg.CptBinaryHierarchy.get_mst(tree), mst)
        self.assertIs(hg.CptMinimumSpanningTree.get_base_graph(mst), base_graph)
        np.testing.assert_array_equal(hg.CptMinimumSpanningTree.get_edge_map(mst), expected_map)
        self.assertEqual(list(zip(*mst.edge_list())), expected_edges)
        self.assertEqual(mst.num_vertices(), base_graph.num_vertices())
        for i, (source, target) in enumerate(expected_edges):
            edge = expected_map[i]
            self.assertEqual({source, target}, {base_graph.sources()[edge], base_graph.targets()[edge]})
            self.assertEqual(tree.lowest_common_ancestor(source, target), tree.num_leaves() + i)
        return mst

    def test_bpt_canonical_input_mst(self):
        for eager in (False, True):
            for return_altitudes in (False, True):
                with self.subTest(eager=eager, return_altitudes=return_altitudes):
                    graph = hg.UndirectedGraph(3)
                    graph.add_edges([0, 0, 1], [1, 2, 2])
                    weights = np.array([10, 1, 2])
                    input_mst = hg.minimum_spanning_tree(graph, weights)
                    result = hg.bpt_canonical(input_mst, weights[input_mst.mst_edge_map],
                                              compute_mst=eager, return_altitudes=return_altitudes)
                    if return_altitudes:
                        tree, altitudes = result
                        np.testing.assert_array_equal(altitudes, [0, 0, 0, 1, 2])
                    else:
                        tree = result
                    self.assertEqual(tree.mst is not None, eager)
                    del input_mst, weights
                    np.testing.assert_array_equal(tree.parents(), [3, 4, 3, 4, 4])
                    mst = self.assert_bpt_mst(tree, graph, [1, 2], [(0, 2), (1, 2)])
                    rebuilt = hg.bpt_canonical(mst, np.array([1, 2]), compute_mst=eager)[0]
                    np.testing.assert_array_equal(rebuilt.parents(), tree.parents())
                    self.assert_bpt_mst(rebuilt, graph, [1, 2], [(0, 2), (1, 2)])

    def test_bpt_canonical_reordered_input_mst_weights_and_ranks(self):
        graph = hg.UndirectedGraph(4)
        graph.add_edges([0, 0, 1, 2, 0], [1, 2, 2, 3, 3])
        # A spanning tree stored in an order different from its base graph and merge order.
        input_mst = hg.subgraph(graph, np.array([3, 1, 2]))
        hg.CptMinimumSpanningTree.link(input_mst, graph, np.array([3, 1, 2]))
        for eager in (False, True):
            for dtype in (np.int32, np.int64, np.float32, np.float64):
                for feature_shape in ((), (2,), (2, 2)):
                    with self.subTest(eager=eager, dtype=dtype, feature_shape=feature_shape):
                        weights = np.broadcast_to(np.array([7, 5, 1], dtype=dtype).reshape(
                            (3,) + (1,) * len(feature_shape)), (3,) + feature_shape).copy()
                        tree, altitudes = hg.bpt_canonical(input_mst, weights, compute_mst=eager)
                        expected_altitudes = np.zeros((7,) + feature_shape, dtype=dtype)
                        expected_altitudes[4:] = weights[[2, 1, 0]]
                        np.testing.assert_array_equal(altitudes, expected_altitudes)
                        self.assertEqual(altitudes.dtype, weights.dtype)
                        np.testing.assert_array_equal(tree.parents(), [5, 4, 4, 6, 5, 6, 6])
                        self.assert_bpt_mst(tree, graph, [2, 1, 3], [(1, 2), (0, 2), (2, 3)])
            order = np.array([2, 0, 1])
            tree, ranks = hg.bpt_canonical(input_mst, sorted_edge_indices=order, compute_mst=eager)
            np.testing.assert_array_equal(ranks, [0, 0, 0, 0, 0, 1, 2])
            self.assert_bpt_mst(tree, graph, [2, 3, 1], [(1, 2), (2, 3), (0, 2)])
            tree, altitudes = hg.bpt_canonical(input_mst, np.array([7, 5, 1]), order, compute_mst=eager)
            np.testing.assert_array_equal(altitudes, [0, 0, 0, 0, 1, 7, 5])
            self.assert_bpt_mst(tree, graph, [2, 3, 1], [(1, 2), (2, 3), (0, 2)])

    def test_bpt_canonical_nested_input_mst(self):
        graph = hg.UndirectedGraph(3)
        graph.add_edges([0, 0, 1], [1, 2, 2])
        first_mst = hg.minimum_spanning_tree(graph, np.array([10, 2, 1]))
        np.testing.assert_array_equal(first_mst.mst_edge_map, [2, 1])
        second_mst = hg.minimum_spanning_tree(first_mst, np.array([2, 1]))
        np.testing.assert_array_equal(second_mst.mst_edge_map, [1, 0])
        for eager in (False, True):
            tree, altitudes = hg.bpt_canonical(second_mst, np.array([1, 2]), compute_mst=eager)
            np.testing.assert_array_equal(altitudes, [0, 0, 0, 1, 2])
            self.assert_bpt_mst(tree, first_mst, [1, 0], [(0, 2), (1, 2)])
            self.assertIs(first_mst.base_graph, graph)
            np.testing.assert_array_equal(first_mst.mst_edge_map, [2, 1])

    def test_bpt_canonical_graph_and_tuple_options(self):
        graph = hg.UndirectedGraph(3)
        graph.add_edges([0, 0, 1], [1, 2, 2])
        weights = np.array([10, 1, 2])
        endpoints = (*graph.edge_list(), graph.num_vertices())
        for graph_input in (graph, endpoints):
            for eager in (False, True):
                for return_altitudes in (False, True):
                    with self.subTest(tuple_input=graph_input is endpoints, eager=eager,
                                      return_altitudes=return_altitudes):
                        result = hg.bpt_canonical(graph_input, weights, return_altitudes=return_altitudes,
                                                  compute_mst=eager)
                        if return_altitudes:
                            tree, altitudes = result
                            np.testing.assert_array_equal(altitudes, [0, 0, 0, 1, 2])
                        else:
                            tree = result
                        np.testing.assert_array_equal(tree.parents(), [3, 4, 3, 4, 4])
                        np.testing.assert_array_equal(tree.mst_edge_map, [1, 2])
                        if graph_input is graph:
                            self.assertEqual(tree.mst is not None, eager)
                            self.assert_bpt_mst(tree, graph, [1, 2], [(0, 2), (1, 2)])
                        else:
                            self.assertIsNone(tree.mst)

    def test_bpt_canonical_singleton_input_mst(self):
        for eager in (False, True):
            graph = hg.UndirectedGraph(1)
            input_mst = hg.minimum_spanning_tree(graph, np.array([], dtype=np.float64))
            tree, altitudes = hg.bpt_canonical(input_mst, np.array([], dtype=np.float64), compute_mst=eager)
            np.testing.assert_array_equal(tree.parents(), [0])
            np.testing.assert_array_equal(altitudes, [0])
            self.assert_bpt_mst(tree, graph, [], [])

    def test_QFZ(self):
        graph = hg.get_4_adjacency_graph((2, 3))

        edge_weights = np.asarray((1, 0, 2, 1, 1, 1, 2))

        tree, altitudes = hg.quasi_flat_zone_hierarchy(graph, edge_weights)

        tref = hg.Tree(np.asarray((6, 7, 8, 6, 7, 8, 7, 9, 9, 9), dtype=np.int64))

        self.assertTrue(hg.test_tree_isomorphism(tree, tref))
        self.assertTrue(np.allclose(altitudes, (0, 0, 0, 0, 0, 0, 0, 1, 1, 2)))

    def test_simplify_tree(self):
        t = TestHierarchyCore.getTree()

        altitudes = np.asarray((0, 0, 0, 0, 0, 1, 2, 2))

        criterion = np.equal(altitudes, altitudes[t.parents()])

        new_tree, node_map = hg.simplify_tree(t, criterion)

        # for reference
        new_altitudes = altitudes[node_map]

        self.assertTrue(new_tree.num_vertices() == 7)

        refp = np.asarray((5, 5, 6, 6, 6, 6, 6))
        self.assertTrue(np.all(refp == new_tree.parents()))

        refnm = np.asarray((0, 1, 2, 3, 4, 5, 7))
        self.assertTrue(np.all(refnm == node_map))

    def test_simplify_tree_with_leaves(self):
        t = hg.Tree((7, 7, 8, 8, 8, 9, 9, 11, 10, 10, 11, 11))

        criterion = np.asarray(
            (False, False, False, True, True, True, True, False, True, False, True, False), dtype=np.bool_)

        new_tree, node_map = hg.simplify_tree(t, criterion, process_leaves=True)

        ref_tree = hg.Tree((4, 4, 5, 5, 5, 5))
        self.assertTrue(hg.test_tree_isomorphism(new_tree, ref_tree))

        self.assertFalse(np.max(criterion[node_map]))

    def test_simplify_tree_with_leaves2(self):
        t = hg.Tree((7, 7, 8, 8, 8, 9, 9, 11, 10, 10, 11, 11))

        criterion = np.zeros(t.num_vertices(), dtype=np.bool_)
        criterion[:8] = True
        new_tree, node_map = hg.simplify_tree(t, criterion, process_leaves=True)

        ref_tree = hg.Tree((2, 2, 3, 3))
        self.assertTrue(hg.test_tree_isomorphism(new_tree, ref_tree))

        self.assertFalse(np.max(criterion[node_map]))

    def test_simplify_tree_with_leaves3(self):
        t = hg.Tree((7, 7, 8, 8, 8, 9, 9, 11, 10, 10, 11, 11))

        criterion = np.zeros(t.num_vertices(), dtype=np.bool_)
        criterion[:9] = True
        new_tree, node_map = hg.simplify_tree(t, criterion, process_leaves=True)

        ref_tree = hg.Tree((1, 2, 2))
        self.assertTrue(hg.test_tree_isomorphism(new_tree, ref_tree))

        self.assertFalse(np.max(criterion[node_map]))

    def test_simplify_tree_propagate_category(self):
        g = hg.get_4_adjacency_implicit_graph((1, 6))
        vertex_values = np.asarray((1, 5, 4, 3, 3, 6), dtype=np.int32)
        tree, altitudes = hg.component_tree_max_tree(g, vertex_values)

        condition = np.asarray((False, False, False, False, False, False, False, True, False, True, False), np.bool_)

        new_tree, node_map = hg.simplify_tree(tree, condition)

        self.assertTrue(np.all(new_tree.parents() == (8, 7, 7, 8, 8, 6, 8, 8, 8)))
        self.assertTrue(np.all(node_map == (0, 1, 2, 3, 4, 5, 6, 8, 10)))
        self.assertTrue(new_tree.category() == hg.TreeCategory.ComponentTree)

        rec = hg.reconstruct_leaf_data(new_tree, altitudes[node_map])
        self.assertTrue(np.all(rec == (1, 4, 4, 1, 1, 6)))

    def test_saliency(self):
        graph = hg.get_4_adjacency_graph((2, 3))
        edge_weights = np.asarray((1, 0, 2, 1, 1, 1, 2))
        sm = hg.saliency(*hg.bpt_canonical(graph, edge_weights))

        self.assertTrue(np.all(sm == edge_weights))

        labels = np.asarray(((1, 2, 3),
                             (1, 4, 5)))
        rag = hg.make_region_adjacency_graph_from_labelisation(graph, labels)
        rag_edge_weights = np.asarray((1, 2, 1, 1, 1, 2))
        sm = hg.saliency(*hg.bpt_canonical(rag, rag_edge_weights))
        self.assertTrue(np.all(sm == edge_weights))

    def test_canonize_tree(self):
        t = TestHierarchyCore.getTree()
        altitudes = np.asarray((0, 0, 0, 0, 0, 1, 2, 2))

        new_tree, new_altitudes, node_map = hg.canonize_hierarchy(t, altitudes, return_node_map=True)

        refp = np.asarray((5, 5, 6, 6, 6, 6, 6))
        self.assertTrue(np.all(refp == new_tree.parents()))

        refa = np.asarray((0, 0, 0, 0, 0, 1, 2))
        self.assertTrue(np.all(refa == new_altitudes))

        refm = np.asarray((0, 1, 2, 3, 4, 5, 7))
        self.assertTrue(np.all(refm == node_map))

    def test_tree_2_binary_tree(self):
        t = hg.Tree((9, 9, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12))

        new_tree, node_map = hg.tree_2_binary_tree(t)

        exp_parents = np.asarray((9, 9, 10, 10, 11, 12, 13, 13, 14, 15, 11, 12, 15, 14, 16, 16, 16), dtype=np.int64)
        exp_node_map = np.asarray((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12), dtype=np.int64)

        self.assertTrue(np.all(new_tree.parents() == exp_parents))
        self.assertTrue(np.all(node_map == exp_node_map))


if __name__ == '__main__':
    unittest.main()
