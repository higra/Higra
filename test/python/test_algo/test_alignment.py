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


class TestAlignement(unittest.TestCase):

    def test_hierarchy_alignement(self):
        g = hg.get_4_adjacency_graph((3, 3))
        fine_labels = np.asarray((0, 1, 2, 3, 4, 2, 3, 4, 2), dtype=np.int32)
       
        aligner = hg.HierarchyAligner.from_labelisation(g, fine_labels)
       
        t = hg.Tree(np.asarray((9, 10, 10, 9, 11, 11, 9, 11, 11, 13, 12, 12, 13, 13), dtype=np.int64))
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2), dtype=np.int32)
        
        sm = aligner.align_hierarchy(t, altitudes)
        sm_k = hg.graph_4_adjacency_2_khalimsky(sm, g, (3, 3))

        sm_coarse = np.asarray((2, 0, 0, 1, 1, 2, 0, 0, 0, 0, 2, 0), dtype=np.int32)
        
        sm2 = aligner.align_hierarchy(g, sm_coarse)
        sm2_k = hg.graph_4_adjacency_2_khalimsky(sm2, g, (3, 3))
        
        sm_k_ref = np.asarray(((0, 2, 0, 1, 0),
                               (0, 2, 1, 1, 0),
                               (0, 2, 0, 0, 0),
                               (0, 2, 0, 0, 0),
                               (0, 2, 0, 0, 0)), dtype=np.int32)

        self.assertTrue(np.all(sm_k == sm_k_ref))
        self.assertTrue(np.all(sm2_k == sm_k_ref))


if __name__ == '__main__':
    unittest.main()
