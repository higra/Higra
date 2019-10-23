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
        mst_edge_map = hg.get_attribute(mst, "mst_edge_map")

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
            (False, False, False, True, True, True, True, False, True, False, True, False), dtype=np.bool)

        new_tree, node_map = hg.simplify_tree(t, criterion, process_leaves=True)

        ref_tree = hg.Tree((4, 4, 5, 5, 5, 5))
        self.assertTrue(hg.test_tree_isomorphism(new_tree, ref_tree))

        self.assertFalse(np.max(criterion[node_map]))

    def test_simplify_tree_propagate_category(self):
        g = hg.get_4_adjacency_implicit_graph((1, 6))
        vertex_values = np.asarray((1, 5, 4, 3, 3, 6), dtype=np.int32)
        tree, altitudes = hg.component_tree_max_tree(g, vertex_values)

        condition = np.asarray((False, False, False, False, False, False, False, True, False, True, False), np.bool)

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
        rag_edge_weights = (1, 2, 1, 1, 1, 2)
        sm = hg.saliency(*hg.bpt_canonical(rag, rag_edge_weights))
        self.assertTrue(np.all(sm == edge_weights))

    def test_canonize_tree(self):
        t = TestHierarchyCore.getTree()
        altitudes = np.asarray((0, 0, 0, 0, 0, 1, 2, 2))

        new_tree, new_altitudes = hg.canonize_hierarchy(t, altitudes)

        refp = np.asarray((5, 5, 6, 6, 6, 6, 6))
        self.assertTrue(np.all(refp == new_tree.parents()))

        refa = np.asarray((0, 0, 0, 0, 0, 1, 2))
        self.assertTrue(np.all(refa == new_altitudes))

    def test_tree_2_binary_tree(self):
        t = hg.Tree((9, 9, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12))

        new_tree, node_map = hg.tree_2_binary_tree(t)

        exp_parents = np.asarray((9, 9, 10, 10, 11, 12, 13, 13, 14, 15, 11, 12, 15, 14, 16, 16, 16), dtype=np.int64)
        exp_node_map = np.asarray((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12), dtype=np.int64)

        self.assertTrue(np.all(new_tree.parents() == exp_parents))
        self.assertTrue(np.all(node_map == exp_node_map))


if __name__ == '__main__':
    unittest.main()
