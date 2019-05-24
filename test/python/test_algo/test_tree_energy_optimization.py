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

    def test_hierarchy_to_optimal_MumfordShah_energy_cut_hierarchy(self):
        # Test strategy:
        # 1) start from a random hierarchy
        # 2) construct the corresponding optimal Mumford-Shah energy cut hierarchy
        # 3) verify that the horizontal cuts of the new hierarchy corresponds to the
        # optimal energy cuts of the first hierarchy obtained from the explicit MF energy
        # and the function labelisation_optimal_cut_from_energy

        shape = (10, 10)
        g = hg.get_4_adjacency_graph(shape)
        np.random.seed(2)
        vertex_weights = np.random.rand(*shape)
        edge_weights = hg.weight_graph(g, vertex_weights, hg.WeightFunction.L1)
        tree1, altitudes1 = hg.watershed_hierarchy_by_area(g, edge_weights)

        tree, altitudes = hg.hierarchy_to_optimal_MumfordShah_energy_cut_hierarchy(tree1, vertex_weights,
                                                                                   approximation_piecewise_linear_function=999999)

        for a in altitudes:
            if a != 0:
                res = False
                cut1 = hg.labelisation_horizontal_cut_from_threshold(tree, altitudes, a)
                # du to numerical issues, and especially as we test critical scale level lambda,
                # we test several very close scale levels
                for margin in [-1e-8, 0, 1e-8]:
                    mfs_energy = hg.attribute_piecewise_constant_Mumford_Shah_energy(tree1, vertex_weights, a + margin)
                    cut2 = hg.labelisation_optimal_cut_from_energy(tree1, mfs_energy)
                    res = res or hg.is_in_bijection(cut1, cut2)
                self.assertTrue(res)

    def test_binary_partition_tree_MumfordShah_energy_scalar(self):
        g = hg.get_4_adjacency_graph((3, 3))

        vertex_values = np.asarray((1, 1, 20, 6, 1, 20, 10, 10, 10))

        tree, altitudes = hg.binary_partition_tree_MumfordShah_energy(
            g, vertex_values)
        ref_parents = (10, 10, 11, 14, 13, 11, 12, 9, 9, 12, 13, 16, 15, 14, 15, 16, 16)
        ref_altitudes = (0., 0., 0.,
                         0., 0., 0.,
                         0., 0., 0.,
                         0., 0., 0.,
                         0., 0., 4.6875, 25.741071, 53.973545)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.allclose(altitudes, ref_altitudes))

    def test_binary_partition_tree_MumfordShah_energy_vectorial(self):
        g = hg.get_4_adjacency_graph((3, 3))

        vertex_values = np.asarray(((1, 2), (1, 2), (20, 30), (6, 7), (1, 2), (20, 30), (10, 12), (10, 12),
                                    (10, 12)))

        tree, altitudes = hg.binary_partition_tree_MumfordShah_energy(
            g, vertex_values)
        ref_parents = (10, 10, 11, 14, 13, 11, 12, 9, 9, 12, 13, 16, 15, 14, 15, 16, 16)
        ref_altitudes = (0., 0., 0.,
                         0., 0., 0.,
                         0., 0., 0.,
                         0., 0., 0.,
                         0., 0., 9.375, 58.553571, 191.121693)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.allclose(altitudes, ref_altitudes))


if __name__ == '__main__':
    unittest.main()
