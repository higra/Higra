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
        graph = hg._get_4_adjacency_graph((1, 2))

        edge_weights = np.asarray((2,))

        tree, altitudes, mst = hg._bpt_canonical(graph, edge_weights)

        self.assertTrue(tree.num_vertices() == 3)
        self.assertTrue(tree.num_edges() == 2)
        self.assertTrue(np.allclose(tree.parents(), (2, 2, 2)))
        self.assertTrue(np.allclose(altitudes, (0, 0, 2)))
        self.assertTrue(mst.num_vertices() == 2)
        self.assertTrue(mst.num_edges() == 1)

    def test_BPT(self):
        graph = hg._get_4_adjacency_graph((2, 3))

        edge_weights = np.asarray((1, 0, 2, 1, 1, 1, 2))

        tree, altitudes, mst = hg._bpt_canonical(graph, edge_weights)
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

    def test_simplifyTree(self):
        t = TestHierarchyCore.getTree()

        altitudes = np.asarray((0, 0, 0, 0, 0, 1, 2, 2))

        criterion = np.equal(altitudes, altitudes[t.parents()])

        new_tree, node_map = hg._simplify_tree(t, criterion)

        # for reference
        new_altitudes = altitudes[node_map]

        self.assertTrue(new_tree.num_vertices() == 7)

        refp = np.asarray((5, 5, 6, 6, 6, 6, 6))
        self.assertTrue(np.all(refp == new_tree.parents()))

        refnm = np.asarray((0, 1, 2, 3, 4, 5, 7))
        self.assertTrue(np.all(refnm == node_map))

    def test_compute_bpt_merge_attribute(self):
        parent = np.asarray((6, 6, 7, 8, 9, 10, 7, 8, 9, 10, 10), dtype=np.int32)
        tree = hg.Tree(parent)
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2), dtype=np.float32)
        attribute = np.asarray((2, 3, 6, 5, 5, 1, 5, 11, 16, 21, 22), dtype=np.float32)

        # ref = (2, 3, 6, 5, 5, 1, 3, 6, 6, 21, 22) extinction
        ref = (0, 0, 0, 0, 0, 0, 2, 3, 5, 5, 1)  # persistence
        merge_attribute = hg.compute_bpt_merge_attribute(tree, attribute, altitudes)
        self.assertTrue(np.allclose(ref, merge_attribute))

if __name__ == '__main__':
    unittest.main()
