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
        sm_k = hg.graph_4_adjacency_2_khalimsky(g, sm, (3, 3))

        sm_coarse = np.asarray((2, 0, 0, 1, 1, 2, 0, 0, 0, 0, 2, 0), dtype=np.int32)

        sm2 = aligner.align_hierarchy(g, sm_coarse)
        sm2_k = hg.graph_4_adjacency_2_khalimsky(g, sm2, (3, 3))

        sm_k_ref = np.asarray(((0, 2, 0, 1, 0),
                               (0, 2, 1, 1, 0),
                               (0, 2, 0, 0, 0),
                               (0, 2, 0, 0, 0),
                               (0, 2, 0, 0, 0)), dtype=np.int32)

        self.assertTrue(np.all(sm_k == sm_k_ref))
        self.assertTrue(np.all(sm2_k == sm_k_ref))

    def test_align_hierarchies(self):
        g = hg.get_4_adjacency_graph((3, 3))
        fine_labels = np.asarray((0, 1, 2, 3, 4, 2, 3, 4, 2), dtype=np.int32)

        t = hg.Tree(np.asarray((9, 10, 10, 9, 11, 11, 9, 11, 11, 13, 12, 12, 13, 13), dtype=np.int64))
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2), dtype=np.int32)

        t2 = hg.Tree(np.asarray((4, 3, 3, 4, 4), dtype=np.int64))
        rag = hg.make_region_adjacency_graph_from_labelisation(g, np.asarray((0, 1, 1,
                                                                              0, 2, 2,
                                                                              0, 2, 2)))
        hg.CptHierarchy.link(t2, rag)
        altitudes2 = np.asarray((0, 0, 0, 1, 2), dtype=np.int32)
        sm_coarse = np.asarray((2, 0, 0, 1, 1, 2, 0, 0, 0, 0, 2, 0), dtype=np.int32)

        sm_coarse_rag = np.asarray((2, 1, 2), dtype=np.int32)

        sm_k_ref = np.asarray(((0, 2, 0, 1, 0),
                               (0, 2, 1, 1, 0),
                               (0, 2, 0, 0, 0),
                               (0, 2, 0, 0, 0),
                               (0, 2, 0, 0, 0)), dtype=np.int32)

        # test only one:
        res = hg.align_hierarchies(g, fine_labels, (t, altitudes))
        res_k = hg.graph_4_adjacency_2_khalimsky(g, res, (3, 3))
        self.assertTrue(np.all(res_k == sm_k_ref))

        res = hg.align_hierarchies(g, fine_labels, (t2, altitudes2))
        res_k = hg.graph_4_adjacency_2_khalimsky(g, res, (3, 3))
        self.assertTrue(np.all(res_k == sm_k_ref))

        res = hg.align_hierarchies(g, fine_labels, (g, sm_coarse))
        res_k = hg.graph_4_adjacency_2_khalimsky(g, res, (3, 3))
        self.assertTrue(np.all(res_k == sm_k_ref))

        res = hg.align_hierarchies(g, fine_labels, (rag, sm_coarse_rag))
        res_k = hg.graph_4_adjacency_2_khalimsky(g, res, (3, 3))
        self.assertTrue(np.all(res_k == sm_k_ref))

        # test several
        res_all = hg.align_hierarchies(g, fine_labels,
                                       ((t2, altitudes2), (g, sm_coarse), (t, altitudes), (rag, sm_coarse_rag)))
        for res in res_all:
            res_k = hg.graph_4_adjacency_2_khalimsky(g, res, (3, 3))
            self.assertTrue(np.all(res_k == sm_k_ref))

if __name__ == '__main__':
    unittest.main()
