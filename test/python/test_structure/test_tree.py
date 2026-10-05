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
import pickle
import numpy as np
import higra as hg


class ExtendedTree(hg.Tree):
    pass


class LegacyTreePickle:
    """Produce the historical reduction tuple without invoking Tree.__reduce__."""

    def __init__(self, tree):
        self.tree = tree

    def __reduce__(self):
        return hg.Tree, (self.tree.parents(),), self.tree.__dict__


class TestTree(unittest.TestCase):

    @staticmethod
    def get_tree():
        parent_relation = np.asarray((5, 5, 6, 6, 6, 7, 7, 7), dtype=np.uint64)
        return hg.Tree(parent_relation)

    def test_size_tree(self):
        t = TestTree.get_tree()

        self.assertTrue(t.category() == hg.TreeCategory.PartitionTree)
        self.assertTrue(t.root() == 7)
        self.assertTrue(t.num_vertices() == 8)
        self.assertTrue(t.num_edges() == 7)
        self.assertTrue(t.num_leaves() == 5)

        self.assertTrue(t.is_leaf(0))
        self.assertTrue(not t.is_leaf(5))
        self.assertTrue(np.all(t.is_leaf((0, 5, 2, 3, 7)) == (True, False, True, True, False)))

        self.assertTrue(t.num_children(6) == 3)
        self.assertTrue(np.all(t.num_children((5, 7, 6)) == (2, 2, 3)))
        self.assertTrue(np.all(t.num_children() == (2, 3, 2)))

        self.assertTrue(t.parent(4) == 6)
        self.assertTrue(np.all(t.parent((0, 5, 2, 3, 7)) == (5, 7, 6, 6, 7)))

    def test_empty_tree(self):
        t = hg.Tree([])

        self.assertTrue(t.category() == hg.TreeCategory.PartitionTree)
        self.assertTrue(t.root() == -1)
        self.assertTrue(t.num_vertices() == 0)
        self.assertTrue(t.num_edges() == 0)
        self.assertTrue(t.num_leaves() == 0)

    def test_dynamic_attributes(self):
        t = TestTree.get_tree()
        t.new_attribute = 42
        self.assertTrue(t.new_attribute == 42)

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
            res.append((t.source(e), t.target(e)))

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
                res.append((e[0], e[1]))
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
                res.append((e[0], e[1]))
            self.assertTrue(res == ref[v])

    def test_edge_index_iterator(self):
        t = TestTree.get_tree()

        ref = [0, 1, 2, 3, 4, 5, 6]
        res = []

        for e in t.edges():
            res.append(t.index(e))

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
            for e in t.out_edges(v):
                res.append(e[2])
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
            for e in t.in_edges(v):
                res.append(e[2])
            self.assertTrue(res == ref[v])

    def test_edge_list(self):
        g = TestTree.get_tree()
        ref_sources = (0, 1, 2, 3, 4, 5, 6)
        ref_targets = (5, 5, 6, 6, 6, 7, 7)

        sources = g.sources()
        self.assertTrue(np.all(ref_sources == sources))

        targets = g.targets()
        self.assertTrue(np.all(ref_targets == targets))

        sources, targets = g.edge_list()
        self.assertTrue(np.all(ref_sources == sources))
        self.assertTrue(np.all(ref_targets == targets))

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

        self.assertTrue(t.child(1, 5) == 1)
        self.assertTrue(np.all(t.child(0, (5, 7, 6)) == (0, 5, 2)))
        self.assertTrue(np.all(t.child(1, (5, 7, 6)) == (1, 6, 3)))

    def test_leaves_iterator(self):
        t = TestTree.get_tree()

        ref = [0, 1, 2, 3, 4]
        self.assertTrue(ref == [l for l in t.leaves()])

    def test_ancestors_iterator(self):
        t = TestTree.get_tree()

        self.assertTrue(np.all([1, 5, 7] == t.ancestors(1)))
        self.assertTrue(np.all([6, 7] == t.ancestors(6)))
        self.assertTrue(np.all([7] == t.ancestors(7)))

    def test_find_region(self):
        tree = hg.Tree((8, 8, 9, 7, 7, 11, 11, 9, 10, 10, 12, 12, 12))

        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 2, 3), dtype=np.int32)
        vertices = np.asarray((0, 0, 0, 2, 2, 9, 9, 12), dtype=np.int64)
        lambdas = np.asarray((2, 3, 4, 1, 2, 2, 3, 3), dtype=np.float64)

        expected_results = np.asarray((0, 10, 12, 2, 9, 9, 10, 12), dtype=np.int64)

        for i in range(vertices.size):
            self.assertTrue(tree.find_region(vertices[i], lambdas[i], altitudes) == expected_results[i])

        self.assertTrue(np.all(tree.find_region(vertices, lambdas, altitudes) == expected_results))

    def test_lowest_common_ancestor_scalar(self):
        t = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7))

        self.assertTrue(t.lowest_common_ancestor(0, 0) == 0)
        self.assertTrue(t.lowest_common_ancestor(3, 3) == 3)
        self.assertTrue(t.lowest_common_ancestor(5, 5) == 5)
        self.assertTrue(t.lowest_common_ancestor(7, 7) == 7)
        self.assertTrue(t.lowest_common_ancestor(0, 1) == 5)
        self.assertTrue(t.lowest_common_ancestor(1, 0) == 5)
        self.assertTrue(t.lowest_common_ancestor(2, 3) == 6)
        self.assertTrue(t.lowest_common_ancestor(2, 4) == 6)
        self.assertTrue(t.lowest_common_ancestor(3, 4) == 6)
        self.assertTrue(t.lowest_common_ancestor(5, 6) == 7)
        self.assertTrue(t.lowest_common_ancestor(0, 2) == 7)
        self.assertTrue(t.lowest_common_ancestor(1, 4) == 7)
        self.assertTrue(t.lowest_common_ancestor(2, 6) == 6)

    def test_lowest_common_ancestor_vectorial(self):
        t = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7))
        v1 = np.asarray((0, 0, 1, 3), dtype=np.int64)
        v2 = np.asarray((0, 3, 0, 0), dtype=np.int64)

        res = t.lowest_common_ancestor(v1, v2)

        ref = np.asarray((0, 7, 5, 7), dtype=np.int64)
        self.assertTrue(np.all(res == ref))

    def test_pickle(self):
        for category in (hg.TreeCategory.PartitionTree, hg.TreeCategory.ComponentTree):
            for tree_class in (hg.Tree, ExtendedTree):
                for protocol in (2, 4, pickle.HIGHEST_PROTOCOL):
                    with self.subTest(category=category, tree_class=tree_class, protocol=protocol):
                        t = tree_class((5, 5, 6, 6, 6, 7, 7, 7), category)
                        hg.set_attribute(t, "test", (1, 2, 3))
                        hg.add_tag(t, "foo")
                        # Exercise serialization after lazy child computation too.
                        t.children(t.root())
                        t2 = pickle.loads(pickle.dumps(t, protocol=protocol))

                        self.assertIs(type(t2), tree_class)
                        self.assertEqual(t2.category(), category)
                        np.testing.assert_array_equal(t.parents(), t2.parents())
                        for v in t.vertices():
                            np.testing.assert_array_equal(t.children(v), t2.children(v))
                        self.assertEqual(hg.get_attribute(t2, "test"), (1, 2, 3))
                        self.assertEqual(t2.test, t.test)
                        self.assertTrue(hg.has_tag(t2, "foo"))

    def test_pickle_hierarchy_reconstruction(self):
        graph = hg.get_4_adjacency_graph((1, 5))
        for category in (hg.TreeCategory.PartitionTree, hg.TreeCategory.ComponentTree):
            for dtype in (np.int32, np.float64):
                for vector in (False, True):
                    with self.subTest(category=category, dtype=dtype, vector=vector):
                        tree = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7), category)
                        hg.CptHierarchy.link(tree, graph)
                        restored = pickle.loads(pickle.dumps(tree))
                        restored_graph = hg.CptHierarchy.get_leaf_graph(restored)
                        self.assertTrue(hg.CptHierarchy.validate(restored))
                        self.assertTrue(hg.CptGridGraph.validate(restored_graph))
                        self.assertEqual(hg.CptGridGraph.get_shape(restored_graph), (1, 5))
                        np.testing.assert_array_equal(restored_graph.sources(), graph.sources())
                        np.testing.assert_array_equal(restored_graph.targets(), graph.targets())

                        altitudes = np.arange(16 if vector else 8, dtype=dtype)
                        if vector:
                            altitudes = altitudes.reshape((8, 2))
                        leaf_nodes = np.arange(tree.num_leaves())
                        if category == hg.TreeCategory.ComponentTree:
                            leaf_nodes = tree.parents()[leaf_nodes]
                        expected = altitudes[leaf_nodes].reshape((1, 5, 2) if vector else (1, 5))
                        for deleted in (None, np.zeros(tree.num_vertices(), dtype=bool)):
                            actual = hg.reconstruct_leaf_data(restored, altitudes, deleted)
                            np.testing.assert_array_equal(actual, expected)
                            np.testing.assert_array_equal(actual, hg.reconstruct_leaf_data(tree, altitudes, deleted))
                            self.assertEqual(actual.dtype, altitudes.dtype)

    def test_pickle_binary_hierarchy_maps(self):
        graph = hg.get_4_adjacency_graph((2, 3))
        tree, _ = hg.bpt_canonical(graph, np.arange(graph.num_edges(), dtype=np.float64))
        restored = pickle.loads(pickle.dumps(tree))
        self.assertEqual(restored.category(), tree.category())
        np.testing.assert_array_equal(restored.parents(), tree.parents())
        self.assertTrue(hg.CptBinaryHierarchy.validate(restored))
        original = hg.CptBinaryHierarchy.construct(tree)
        result = hg.CptBinaryHierarchy.construct(restored)
        np.testing.assert_array_equal(result["mst_edge_map"], original["mst_edge_map"])
        np.testing.assert_array_equal(result["mst"].sources(), original["mst"].sources())
        np.testing.assert_array_equal(result["mst"].targets(), original["mst"].targets())
        self.assertEqual(hg.CptGridGraph.get_shape(result["leaf_graph"]), (2, 3))
        self.assertTrue(hg.CptMinimumSpanningTree.validate(result["mst"]))
        self.assertIs(hg.CptMinimumSpanningTree.get_base_graph(result["mst"]), result["leaf_graph"])
        np.testing.assert_array_equal(hg.CptMinimumSpanningTree.get_edge_map(result["mst"]), result["mst_edge_map"])

    def test_pickle_legacy_category_default(self):
        for category in (hg.TreeCategory.PartitionTree, hg.TreeCategory.ComponentTree):
            for protocol in (2, 4, pickle.HIGHEST_PROTOCOL):
                with self.subTest(category=category, protocol=protocol):
                    tree = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7), category)
                    hg.set_attribute(tree, "test", (1, 2, 3))
                    hg.add_tag(tree, "foo")
                    hg.CptHierarchy.link(tree, hg.get_4_adjacency_graph((1, 5)))
                    # Trusted, controlled old-format pickle: category was never stored.
                    restored = pickle.loads(pickle.dumps(LegacyTreePickle(tree), protocol=protocol))
                    self.assertEqual(restored.category(), hg.TreeCategory.PartitionTree)
                    np.testing.assert_array_equal(restored.parents(), tree.parents())
                    for v in tree.vertices():
                        np.testing.assert_array_equal(restored.children(v), tree.children(v))
                    self.assertEqual(restored.test, (1, 2, 3))
                    self.assertTrue(hg.has_tag(restored, "foo"))
                    self.assertTrue(hg.CptHierarchy.validate(restored))
                    self.assertEqual(hg.CptGridGraph.get_shape(hg.CptHierarchy.get_leaf_graph(restored)), (1, 5))

    def test_sub_tree(self):
        tree = hg.Tree(np.asarray((8, 8, 9, 9, 10, 10, 11, 13, 12, 12, 11, 13, 14, 14, 14)))

        # full tree
        sub_tree, node_map = tree.sub_tree(14)
        self.assertTrue(np.all(tree.parents() == sub_tree.parents()))
        self.assertTrue(np.all(np.arange(tree.num_vertices()) == node_map))

        # normal
        sub_tree, node_map = tree.sub_tree(13)
        self.assertTrue(np.all(sub_tree.parents() == (4, 4, 5, 6, 5, 6, 6)))
        self.assertTrue(np.all(node_map == (4, 5, 6, 7, 10, 11, 13)))

        # leaf
        sub_tree, node_map = tree.sub_tree(3)
        self.assertTrue(np.all(sub_tree.parents() == (0,)))
        self.assertTrue(np.all(node_map == (3,)))

    def test_tree_2_undirected_graph(self):
        tree = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7))

        g = tree.to_undirected_graph()

        self.assertTrue(isinstance(g, hg.UndirectedGraph))
        self.assertTrue(g.num_vertices() == tree.num_vertices())
        self.assertTrue(g.num_edges() == tree.num_edges())
        self.assertTrue(np.all(g.sources() == tree.sources()))
        self.assertTrue(np.all(g.targets() == tree.targets()))

        g = tree.to_undirected_graph(include_leaves=False)

        self.assertTrue(isinstance(g, hg.UndirectedGraph))
        self.assertTrue(g.num_vertices() == tree.num_vertices() - tree.num_leaves())
        self.assertTrue(g.num_edges() == tree.num_edges() - tree.num_leaves())
        self.assertTrue(np.all(g.sources() == tree.sources()[tree.num_leaves():] - tree.num_leaves()))
        self.assertTrue(np.all(g.targets() == tree.targets()[tree.num_leaves():] - tree.num_leaves()))


if __name__ == '__main__':
    unittest.main()
