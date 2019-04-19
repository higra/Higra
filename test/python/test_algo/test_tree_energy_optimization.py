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


class TestTreeEnergyOptimization(unittest.TestCase):

    def test_labelisation_optimal_cut_from_energy(self):
        t = hg.Tree((8, 8, 9, 7, 7, 11, 11, 9, 10, 10, 12, 12, 12))
        energy_attribute = np.asarray((2, 1, 3, 2, 1, 1, 1, 2, 2, 4, 10, 5, 20))

        res = hg.labelisation_optimal_cut_from_energy(t, energy_attribute)

        ref = np.asarray((0, 0, 1, 1, 1, 2, 3))
        self.assertTrue(hg.is_in_bijection(res, ref))

    def test_hierarchy_to_optimal_energy_cut_hierarchy(self):
        t = hg.Tree((8, 8, 9, 7, 7, 11, 11, 9, 10, 10, 12, 12, 12))
        data_fidelity_attribute = np.asarray((1, 1, 1, 1, 1, 1, 1, 4, 5, 10, 15, 25, 45))
        regularization_attribute = np.asarray((4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 10, 4, 12))

        tree, altitudes = hg.hierarchy_to_optimal_energy_cut_hierarchy(t, data_fidelity_attribute,
                                                                       regularization_attribute)

        ref_parents = np.asarray((8, 8, 9, 7, 7, 10, 10, 9, 9, 10, 10))
        ref_altitudes = np.asarray((0., 0., 0., 0., 0., 0., 0., 0.5, 0.75, 2.5, 14.0 / 3))
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.allclose(altitudes, ref_altitudes))


if __name__ == '__main__':
    unittest.main()
