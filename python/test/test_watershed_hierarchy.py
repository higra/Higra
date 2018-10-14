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


class TestWatershedHierarchy(unittest.TestCase):

    def test_watershed_hierarchy_by_attribute(self):
        g = hg.get_4_adjacency_graph((1, 19))
        edge_weights = np.asarray((0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0))
        # watershed hierarchy by area...
        res = hg._watershed_hierarchy_by_attribute(g, edge_weights,
                                                  lambda tree, altitudes:
                                                  hg.accumulate_sequential(tree, np.ones((tree.num_leaves(),)), hg.Accumulators.sum))
        t = res.tree()
        altitudes = res.node_altitude()

        ref_parents = np.asarray((
            19, 19, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 23, 23, 23, 24, 24, 25,
            26, 26, 25, 27, 27, 27), dtype=np.int64)
        ref_tree = hg.Tree(ref_parents)
        ref_altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 5))

        self.assertTrue(hg.test_tree_isomorphism(t, ref_tree))
        self.assertTrue(np.allclose(altitudes, ref_altitudes))



if __name__ == '__main__':
    unittest.main()
