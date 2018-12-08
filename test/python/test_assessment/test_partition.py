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


if __name__ == '__main__':
    unittest.main()
