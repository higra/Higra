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


class TestFragmentationCurve(unittest.TestCase):

    def test_assess_fragmentation_curve_BCE_optimal_cut(self):
        t = hg.Tree((8, 8, 9, 9, 10, 10, 11, 13, 12, 12, 11, 13, 14, 14, 14))
        ground_truth = np.asarray((0, 0, 1, 1, 1, 2, 2, 2), dtype=np.int32)

        assesser = hg.make_assesser_fragmentation_optimal_cut(t, ground_truth, hg.OptimalCutMeasure.BCE)

        self.assertTrue(assesser.optimal_number_of_regions() == 3)
        self.assertTrue(assesser.number_of_region_ground_truth() == 3)
        self.assertTrue(np.isclose(assesser.optimal_score(), (2 + 4.0 / 3 + 2.5) / t.num_leaves()))

        res_k, res_scores = assesser.fragmentation_curve(False)

        ref_scores = np.asarray((2.75, 4.5, 2 + 4.0 / 3 + 2.5, 2 + 4.0 / 3 + 2, 2 + 4.0 / 3 + 4.0 / 3,
                                 2 + 4.0 / 3 + 4.0 / 3, 4, 3))
        ref_k = np.asarray((1, 2, 3, 4, 5, 6, 7, 8), dtype=np.int32)

        self.assertTrue(np.all(res_scores == (ref_scores / t.num_leaves())))
        self.assertTrue(np.all(res_k == ref_k))

    def test_assess_fragmentation_curve_BCE_optimal_cut_on_rag(self):
        vertex_map = np.asarray((0, 0, 1, 1, 2, 2, 3, 4), dtype=np.int64)
        g = hg.get_4_adjacency_graph((1, 8))
        hg.CptRegionAdjacencyGraph.link(g, None, vertex_map, np.ones((7, )))
        t = hg.Tree((6, 6, 5, 5, 7, 7, 8, 8, 8))
        hg.CptHierarchy.link(t, g)
        ground_truth = np.asarray((0, 0, 1, 1, 1, 2, 2, 2), dtype=np.int32)

        assesser = hg.make_assesser_fragmentation_optimal_cut(t, ground_truth, hg.OptimalCutMeasure.BCE)

        self.assertTrue(assesser.optimal_number_of_regions() == 3)
        self.assertTrue(assesser.number_of_region_ground_truth() == 3)
        self.assertTrue(np.isclose(assesser.optimal_score(), (2 + 4.0 / 3 + 2.5) / t.num_leaves()))

        res_k, res_scores = assesser.fragmentation_curve(False)

        ref_scores = np.asarray((2.75, 4.5, 2 + 4.0 / 3 + 2.5, 2 + 4.0 / 3 + 2, 2 + 4.0 / 3 + 4.0 / 3))
        ref_k = np.asarray((1, 2, 3, 4, 5), dtype=np.int32)

        self.assertTrue(np.all(res_scores == (ref_scores / t.num_leaves())))
        self.assertTrue(np.all(res_k == ref_k))

    def test_assess_optimal_partitions_BCE_optimal_cut(self):
        t = hg.Tree((8, 8, 9, 9, 10, 10, 11, 13, 12, 12, 11, 13, 14, 14, 14))
        ground_truth = np.asarray((0, 0, 1, 1, 1, 2, 2, 2), dtype=np.int32)

        assesser = hg.make_assesser_fragmentation_optimal_cut(t, ground_truth, hg.OptimalCutMeasure.BCE)

        optimal_partitions = [np.asarray((0, 0, 0, 0, 0, 0, 0, 0), dtype=np.int32),
                              np.asarray((0, 0, 0, 0, 1, 1, 1, 1), dtype=np.int32),
                              np.asarray((0, 0, 1, 1, 2, 2, 2, 2), dtype=np.int32),
                              np.asarray((0, 0, 1, 1, 2, 2, 2, 3), dtype=np.int32),
                              np.asarray((0, 0, 1, 1, 2, 2, 3, 4), dtype=np.int32),
                              np.asarray((0, 0, 1, 1, 2, 3, 4, 5), dtype=np.int32),
                              np.asarray((0, 0, 1, 2, 3, 4, 5, 6), dtype=np.int32),
                              np.asarray((0, 1, 2, 3, 4, 5, 6, 7), dtype=np.int32)]

        self.assertTrue(hg.is_in_bijection(optimal_partitions[2], assesser.optimal_partition()))

        for i in range(len(optimal_partitions)):
            self.assertTrue(hg.is_in_bijection(optimal_partitions[i], assesser.optimal_partition(i + 1)))


if __name__ == '__main__':
    unittest.main()
