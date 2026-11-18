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
        self.assertTrue(np.isclose(assesser.optimal_score(), (2 + 4.0 / 3 + 2.5) / t.num_leaves()))

        res = assesser.fragmentation_curve()
        res_k = res.num_regions()
        res_scores = res.scores()

        ref_scores = np.asarray((2.75, 4.5, 2 + 4.0 / 3 + 2.5, 2 + 4.0 / 3 + 2, 2 + 4.0 / 3 + 4.0 / 3,
                                 2 + 4.0 / 3 + 4.0 / 3, 4, 3))
        ref_k = np.asarray((1, 2, 3, 4, 5, 6, 7, 8), dtype=np.int32)

        self.assertTrue(np.all(res_scores == (ref_scores / t.num_leaves())))
        self.assertTrue(np.all(res_k == ref_k))

        res = hg.assess_fragmentation_optimal_cut(t, ground_truth, hg.OptimalCutMeasure.BCE)
        res_k = res.num_regions()
        res_scores = res.scores()
        self.assertTrue(np.all(res_scores == (ref_scores / t.num_leaves())))
        self.assertTrue(np.all(res_k == ref_k))

    def test_assess_fragmentation_curve_BCE_optimal_cut_on_rag(self):
        vertex_map = np.asarray((0, 0, 1, 1, 2, 2, 3, 4), dtype=np.int64)
        g = hg.get_4_adjacency_graph((1, 8))
        hg.CptRegionAdjacencyGraph.link(g, None, vertex_map, np.ones((7,)))
        t = hg.Tree((6, 6, 5, 5, 7, 7, 8, 8, 8))
        hg.CptHierarchy.link(t, g)
        ground_truth = np.asarray((0, 0, 1, 1, 1, 2, 2, 2), dtype=np.int32)

        assesser = hg.make_assesser_fragmentation_optimal_cut(t, ground_truth, hg.OptimalCutMeasure.BCE)

        self.assertTrue(assesser.optimal_number_of_regions() == 3)
        self.assertTrue(np.isclose(assesser.optimal_score(), (2 + 4.0 / 3 + 2.5) / t.num_leaves()))

        res = assesser.fragmentation_curve()
        res_k = res.num_regions()
        res_scores = res.scores()

        ref_scores = np.asarray((2.75, 4.5, 2 + 4.0 / 3 + 2.5, 2 + 4.0 / 3 + 2, 2 + 4.0 / 3 + 4.0 / 3))
        ref_k = np.asarray((1, 2, 3, 4, 5), dtype=np.int32)

        self.assertTrue(np.allclose(res_scores, (ref_scores / t.num_leaves())))
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

    def test_assess_fragmentation_curve_DHamming_horizontal_cut(self):
        tree = hg.Tree((11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18))

        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3))
        ground_truth = np.asarray((0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2), dtype=np.int16)

        res = hg.assess_fragmentation_horizontal_cut(tree,
                                                     altitudes,
                                                     ground_truth,
                                                     hg.PartitionMeasure.DHamming)
        res_scores = res.scores()
        res_k = res.num_regions()

        ref_scores = np.asarray((4.0, 8.0, 9.0, 10.0))
        ref_k = np.asarray((1, 3, 4, 9))

        self.assertTrue(np.allclose(res_scores, (ref_scores / tree.num_leaves())))
        self.assertTrue(np.allclose(res_k, ref_k))

    def test_assess_fragmentation_curve_all_measures_horizontal_cut(self):
        tree = hg.Tree((11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18))
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3))
        ground_truth = np.asarray((0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2), dtype=np.int16)
        # same partition with non contiguous labels (label 2 is not used)
        ground_truth_gap = np.asarray((0, 0, 0, 0, 3, 3, 3, 1, 1, 1, 1), dtype=np.int16)

        ref_k = np.asarray((1, 3, 4, 9))
        ref_scores = {hg.PartitionMeasure.BCE: (3.727272727272727, 5.666666666666666, 6.166666666666666, 3.5),
                      hg.PartitionMeasure.DHamming: (4, 8, 9, 10),
                      hg.PartitionMeasure.DCovering: (4, 6.199999999999999, 5.95, 3.366666666666667)}

        for measure, scores in ref_scores.items():
            for gt in (ground_truth, ground_truth_gap):
                res = hg.assess_fragmentation_horizontal_cut(tree, altitudes, gt, measure)
                self.assertTrue(np.allclose(res.scores() * tree.num_leaves(), scores, rtol=1e-14))
                self.assertTrue(np.all(res.num_regions() == ref_k))
                self.assertTrue(res.num_regions_ground_truth() == 3)
                self.assertTrue(np.allclose(res.num_regions_normalized(), ref_k / 3.0, rtol=1e-15))

            res = hg.assess_fragmentation_horizontal_cut(tree, altitudes, ground_truth, measure, max_regions=4)
            self.assertTrue(np.all(res.num_regions() == ref_k[:3]))
            self.assertTrue(np.allclose(res.scores() * tree.num_leaves(), scores[:3], rtol=1e-14))

    def test_assess_fragmentation_curve_all_measures_optimal_cut_non_contiguous_labels(self):
        t = hg.Tree((8, 8, 9, 9, 10, 10, 11, 13, 12, 12, 11, 13, 14, 14, 14))
        # label 2 is not used
        ground_truth = np.asarray((0, 0, 3, 3, 3, 1, 1, 1), dtype=np.int32)

        ref_k = np.arange(1, 9)
        refs = {hg.OptimalCutMeasure.BCE: ((2.75, 4.5, 5.833333333333333, 5.333333333333333, 4.666666666666666,
                                            4.666666666666666, 4, 3), 3),
                hg.OptimalCutMeasure.DHamming: ((3, 5, 7, 7, 7, 8, 8, 8), 6),
                hg.OptimalCutMeasure.DCovering: ((3, 5, 6.333333333333333, 5.666666666666666, 4.666666666666666,
                                                 4.666666666666666, 4, 3), 3)}

        for measure, (scores, optimal_k) in refs.items():
            assesser = hg.make_assesser_fragmentation_optimal_cut(t, ground_truth, measure)
            res = assesser.fragmentation_curve()
            self.assertTrue(np.allclose(res.scores() * t.num_leaves(), scores, rtol=1e-14))
            self.assertTrue(np.all(res.num_regions() == ref_k))
            self.assertTrue(res.num_regions_ground_truth() == 3)
            self.assertTrue(np.allclose(res.num_regions_normalized(), ref_k / 3.0, rtol=1e-15))
            self.assertTrue(assesser.optimal_number_of_regions() == optimal_k)
            self.assertTrue(np.isclose(assesser.optimal_score() * t.num_leaves(), np.max(scores), rtol=1e-14))


if __name__ == '__main__':
    unittest.main()
