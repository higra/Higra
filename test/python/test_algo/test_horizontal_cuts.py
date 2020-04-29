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


class TestHorizontalCuts(unittest.TestCase):

    def test_horizontal_cut_explorer_indexed(self):
        tree = hg.Tree((11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18))
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3))

        hch = hg.HorizontalCutExplorer(tree, altitudes)

        self.assertTrue(hch.num_cuts() == 4)
        cut_nodes = (
            (18,),
            (17, 13, 14),
            (11, 16, 13, 14),
            (0, 1, 2, 3, 4, 5, 13, 9, 10)
        )
        alt_cuts = (3, 2, 1, 0)

        for i in range(hch.num_cuts()):
            c = hch.horizontal_cut_from_index(i)
            self.assertTrue(np.all(np.sort(c.nodes()) == np.sort(cut_nodes[i])))
            self.assertTrue(c.altitude() == alt_cuts[i])

    def test_horizontal_cut_explorer_assert(self):
        tree = hg.Tree(np.asarray((5, 5, 6, 6, 7, 7, 7, 7)))
        altitudes = np.asarray((0, 0, 1, 0, 0, 2, 1, 1))
        with self.assertRaises(ValueError):
            hch = hg.HorizontalCutExplorer(tree, altitudes)

    def test_horizontal_cut_explorer_altitudes(self):
        tree = hg.Tree((11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18))
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3))

        hch = hg.HorizontalCutExplorer(tree, altitudes)

        self.assertTrue(hch.num_cuts() == 4)
        cut_nodes = (
            (18,),
            (17, 13, 14),
            (11, 16, 13, 14),
            (0, 1, 2, 3, 4, 5, 13, 9, 10)
        )
        alt_cuts = (3, 2, 1, 0)
        for i in range(hch.num_cuts()):
            c = hch.horizontal_cut_from_altitude(alt_cuts[i])
            self.assertTrue(np.all(np.sort(c.nodes()) == np.sort(cut_nodes[i])))
            self.assertTrue(c.altitude() == alt_cuts[i])

    def test_horizontal_cut_explorer_num_regions(self):
        tree = hg.Tree((11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18))
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3))

        hch = hg.HorizontalCutExplorer(tree, altitudes)

        self.assertTrue(hch.num_cuts() == 4)
        cut_nodes = (
            (18,),
            (17, 13, 14),
            (11, 16, 13, 14),
            (0, 1, 2, 3, 4, 5, 13, 9, 10)
        )
        k_cuts = (1, 3, 4, 9)
        for i in range(hch.num_cuts()):
            c = hch.horizontal_cut_from_num_regions(k_cuts[i])
            self.assertTrue(np.all(np.sort(c.nodes()) == np.sort(cut_nodes[i])))

        # cuts with at least the given number of regions
        k_cuts = (1, 2, 4, 5)
        for i in range(hch.num_cuts()):
            c = hch.horizontal_cut_from_num_regions(k_cuts[i])
            self.assertTrue(np.all(np.sort(c.nodes()) == np.sort(cut_nodes[i])))

        # cuts with at most the given number of regions
        k_cuts = (2, 3, 8, 20)
        for i in range(hch.num_cuts()):
            c = hch.horizontal_cut_from_num_regions(k_cuts[i], False)
            self.assertTrue(np.all(np.sort(c.nodes()) == np.sort(cut_nodes[i])))

    def test_horizontal_cut_nodes(self):
        g = hg.get_4_adjacency_graph((1, 11))
        tree = hg.Tree((11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18))
        hg.CptHierarchy.link(tree, g)
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3))

        hch = hg.HorizontalCutExplorer(tree, altitudes)

        cut_nodes = (
            (18,),
            (17, 13, 14),
            (11, 16, 13, 14),
            (0, 1, 2, 3, 4, 5, 13, 9, 10)
        )

        c = hch.horizontal_cut_from_num_regions(3)

        lbls = c.labelisation_leaves(tree)
        ref_lbls = (17, 17, 17, 17, 17, 17, 13, 13, 13, 14, 14)
        self.assertTrue(np.all(lbls == ref_lbls))

        rec = c.reconstruct_leaf_data(tree, altitudes)
        ref_rec = (2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1)
        self.assertTrue(np.all(rec == ref_rec))

        cut = c.graph_cut(tree)
        ref_cut = (0, 0, 0, 0, 0, 1, 0, 0, 1, 0)
        self.assertTrue(np.all(cut == ref_cut))

    def test_horizontal_cut_nodes_delinearization(self):
        g = hg.get_4_adjacency_graph((2, 2))
        tree = hg.Tree((4, 4, 5, 6, 5, 6, 6))
        hg.CptHierarchy.link(tree, g)
        altitudes = np.asarray((0, 0, 0, 0, 1, 2, 3))

        hch = hg.HorizontalCutExplorer(tree, altitudes)

        c = hch.horizontal_cut_from_num_regions(3)

        lbls = c.labelisation_leaves(tree)
        self.assertTrue(np.all(lbls.shape == (2, 2)))
        ref_lbls = np.array((0, 0, 1, 2))
        self.assertTrue(hg.is_in_bijection(lbls[:], ref_lbls))

        vweights = c.reconstruct_leaf_data(tree, altitudes)
        ref_vweights = np.array(((1, 1), (0, 0)))
        self.assertTrue(np.all(vweights == ref_vweights))

    def test_horizontal_cut_nodes_rag_delinearization(self):
        g = hg.get_4_adjacency_graph((2, 2))
        labels = np.array((0, 0, 1, 2))
        rag = hg.make_region_adjacency_graph_from_labelisation(g, labels)
        tree = hg.Tree((3, 3, 4, 4, 4))
        hg.CptHierarchy.link(tree, rag)
        altitudes = np.asarray((0, 0, 0, 1, 2))

        hch = hg.HorizontalCutExplorer(tree, altitudes)

        c = hch.horizontal_cut_from_num_regions(2)

        lbls = c.labelisation_leaves(tree)
        self.assertTrue(np.all(lbls.shape == (2, 2)))
        ref_lbls = np.array((0, 0, 0, 1))
        self.assertTrue(hg.is_in_bijection(lbls[:], ref_lbls))

        vweights = c.reconstruct_leaf_data(tree, altitudes)
        ref_vweights = np.array(((1, 1), (1, 0)))
        self.assertTrue(np.all(vweights == ref_vweights))


if __name__ == '__main__':
    unittest.main()
