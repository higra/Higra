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

    def test_assess_partition(self):
        candidate = np.asarray((0, 0, 0, 1, 1, 1, 2, 2, 2), dtype=np.int32)
        gt1 = np.asarray((0, 0, 1, 1, 1, 2, 2, 3, 3), dtype=np.int32)
        gt2 = np.asarray((0, 0, 0, 0, 1, 1, 1, 1, 1), dtype=np.int32)

        dh1 = hg.assess_partition(candidate, gt1, hg.PartitionMeasure.DHamming)
        s1 = 6.0 / 9
        self.assertTrue(s1 == dh1)

        dh2 = hg.assess_partition(candidate, gt2, hg.PartitionMeasure.DHamming)
        s2 = 8.0 / 9
        self.assertTrue(s2 == dh2)

        dh = hg.assess_partition(candidate, np.stack((gt1, gt2)), hg.PartitionMeasure.DHamming)
        self.assertTrue(np.isclose((s1 + s2) / 2.0, dh))

    def test_assess_partition_measures_non_contiguous_labels(self):
        candidate = np.asarray((0, 0, 1, 1, 1, 2, 2, 0, 2, 1, 1, 2), dtype=np.int32)
        # labels 0 and 2 are not used
        gt1 = np.asarray((1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 1, 1), dtype=np.int32)
        gt2 = np.asarray((0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2), dtype=np.int32)

        expected = {hg.PartitionMeasure.BCE: (0.3652777777777778, 0.4666666666666666, 0.4159722222222222),
                    hg.PartitionMeasure.DHamming: (0.5, 0.75, 0.625),
                    hg.PartitionMeasure.DCovering: (0.3134920634920635, 0.4523809523809524, 0.3829365079365079)}

        for measure, (ref1, ref2, ref12) in expected.items():
            for dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64):
                c = candidate.astype(dtype)
                self.assertTrue(np.isclose(hg.assess_partition(c, gt1.astype(dtype), measure), ref1, rtol=1e-14))
                self.assertTrue(np.isclose(hg.assess_partition(c, gt2.astype(dtype), measure), ref2, rtol=1e-14))
                self.assertTrue(np.isclose(hg.assess_partition(c, np.stack((gt1, gt2)).astype(dtype), measure),
                                           ref12, rtol=1e-14))
            # 2d labelisations and a stack of 2d ground truths
            self.assertTrue(np.isclose(hg.assess_partition(candidate.reshape(3, 4),
                                                           np.stack((gt1, gt2)).reshape(2, 3, 4), measure),
                                       ref12, rtol=1e-14))
            # candidate with unused labels (DHamming and DCovering only: unused candidate labels give NaN with BCE)
            if measure != hg.PartitionMeasure.BCE:
                sparse_candidate = np.asarray((0, 0, 2, 2, 2, 5, 5, 0, 5, 2, 2, 5), dtype=np.int32)
                self.assertTrue(np.isclose(hg.assess_partition(sparse_candidate, gt2, measure), ref2, rtol=1e-14))


    def test_assess_partition_non_contiguous_labels(self):
        candidate = np.asarray((0, 0, 2, 2), dtype=np.int32)
        ground_truth = np.asarray((0, 0, 1, 1), dtype=np.int32)
        ground_truths = np.asarray(((0, 0, 1, 1), (0, 0, 3, 3)), dtype=np.int32)
        for measure in (hg.PartitionMeasure.BCE, hg.PartitionMeasure.DHamming, hg.PartitionMeasure.DCovering):
            self.assertTrue(hg.assess_partition(candidate, ground_truth, measure) == 1)
            self.assertTrue(hg.assess_partition(ground_truth, candidate, measure) == 1)
            self.assertTrue(hg.assess_partition(candidate, ground_truths, measure) == 1)

    def test_assess_partition_empty(self):
        empty = np.zeros((0,), dtype=np.int32)
        candidate = np.asarray((0, 1), dtype=np.int64)
        no_ground_truth = np.empty((0, 2), dtype=np.int64)
        for measure in (hg.PartitionMeasure.BCE, hg.PartitionMeasure.DHamming, hg.PartitionMeasure.DCovering):
            with self.assertRaises(RuntimeError):
                hg.assess_partition(empty, empty, measure)
            with self.assertRaises(RuntimeError):
                hg.assess_partition(candidate, no_ground_truth, measure)

if __name__ == '__main__':
    unittest.main()
