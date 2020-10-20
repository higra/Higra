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
import higra as hg
import numpy as np


class TestAlgorithmTree(unittest.TestCase):

    def test_reconstruct_leaf_data(self):
        tree = hg.Tree(np.asarray((5, 5, 6, 6, 6, 7, 7, 7)))

        input = np.asarray(((1, 8),
                            (2, 7),
                            (3, 6),
                            (4, 5),
                            (5, 4),
                            (6, 3),
                            (7, 2),
                            (8, 1)), dtype=np.int32)

        condition = np.asarray((True, False, True, False, True, True, False, False), np.bool_)

        output = hg.reconstruct_leaf_data(tree, input, condition)
        ref = np.asarray(((8, 1),
                          (2, 7),
                          (7, 2),
                          (4, 5),
                          (7, 2)), dtype=np.int32)

        self.assertTrue(np.all(ref == output))

    def test_reconstruct_leaf_data_component_tree(self):
        g = hg.get_4_adjacency_implicit_graph((1, 6))
        vertex_values = np.asarray((1, 5, 4, 3, 3, 6), dtype=np.int32)
        tree, altitudes = hg.component_tree_max_tree(g, vertex_values)

        condition = np.asarray((True, False, True, False, True, True, False, True, False, True, False), np.bool_)

        output = hg.reconstruct_leaf_data(tree, altitudes, condition)
        ref = np.asarray((1, 4, 4, 1, 1, 6), dtype=np.int32)

        self.assertTrue(np.all(ref == output))

    def test_reconstruct_leaf_data_default(self):
        tree = hg.Tree(np.asarray((5, 5, 6, 6, 6, 7, 7, 7)))

        input = np.asarray(((1, 8),
                            (2, 7),
                            (3, 6),
                            (4, 5),
                            (5, 4),
                            (6, 3),
                            (7, 2),
                            (8, 1)), dtype=np.int32)

        output = hg.reconstruct_leaf_data(tree, input)
        ref = np.asarray(((1, 8),
                          (2, 7),
                          (3, 6),
                          (4, 5),
                          (5, 4)), dtype=np.int32)

        self.assertTrue(np.all(ref == output))

    def test_reconstruct_leaf_data_component_tree_default(self):
        g = hg.get_4_adjacency_implicit_graph((1, 6))
        vertex_values = np.asarray((1, 5, 4, 3, 3, 6), dtype=np.int32)
        tree, altitudes = hg.component_tree_max_tree(g, vertex_values)

        area = hg.attribute_area(tree)
        output = hg.reconstruct_leaf_data(tree, area)
        ref = np.asarray((6, 1, 2, 5, 5, 1), dtype=np.int32)

        self.assertTrue(np.all(ref == output))

    def test_labelisation_horizontal_cut(self):
        tree = hg.Tree(np.asarray((5, 5, 6, 6, 6, 7, 7, 7)))

        altitudes = np.asarray((0, 0, 0, 0, 0, 0.5, 0, 0.7), dtype=np.double)

        ref_t0 = np.asarray((1, 2, 3, 3, 3), dtype=np.int32)
        ref_t1 = np.asarray((1, 1, 2, 2, 2), dtype=np.int32)
        ref_t2 = np.asarray((1, 1, 1, 1, 1), dtype=np.int32)

        output_t0 = hg.labelisation_horizontal_cut_from_threshold(tree, altitudes, 0)
        output_t1 = hg.labelisation_horizontal_cut_from_threshold(tree, altitudes, 0.5)
        output_t2 = hg.labelisation_horizontal_cut_from_threshold(tree, altitudes, 0.7)

        self.assertTrue(hg.is_in_bijection(ref_t0, output_t0))
        self.assertTrue(hg.is_in_bijection(ref_t1, output_t1))
        self.assertTrue(hg.is_in_bijection(ref_t2, output_t2))

    def test_labelisation_horizontal_cut_num_regions(self):
        g = hg.get_4_adjacency_graph((1, 11))
        tree = hg.Tree((11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18))
        hg.CptHierarchy.link(tree, g)
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3))

        ref_labels = (
            np.asarray((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
            np.asarray((1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3)),
            np.asarray((0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3)),
            np.asarray((0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 8))
        )

        k_cuts = (1, 3, 4, 9)
        for i in range(4):
            labels = hg.labelisation_horizontal_cut_from_num_regions(tree, altitudes, k_cuts[i])
            self.assertTrue(hg.is_in_bijection(labels, ref_labels[i]))

        # cuts with at least the given number of regions
        k_cuts = (1, 2, 4, 5)
        for i in range(4):
            labels = hg.labelisation_horizontal_cut_from_num_regions(tree, altitudes, k_cuts[i])
            self.assertTrue(hg.is_in_bijection(labels, ref_labels[i]))

        # cuts with at most the given number of regions
        k_cuts = (2, 3, 8, 20)
        for i in range(4):
            labels = hg.labelisation_horizontal_cut_from_num_regions(tree, altitudes, k_cuts[i], "at_most")
            self.assertTrue(hg.is_in_bijection(labels, ref_labels[i]))

    def test_labelisation_hierarchy_supervertices(self):
        tree = hg.Tree(np.asarray((5, 5, 6, 6, 6, 7, 7, 7)))

        altitudes = np.asarray((0, 0, 0, 0, 0, 0.5, 0, 0.7), dtype=np.double)

        ref = np.asarray((0, 1, 2, 2, 2), dtype=np.int32)

        output = hg.labelisation_hierarchy_supervertices(tree, altitudes)

        self.assertTrue(hg.is_in_bijection(ref, output))
        self.assertTrue(np.amax(output) == 2)
        self.assertTrue(np.amin(output) == 0)

    def test_tree_isomorphism(self):
        t1 = hg.Tree(np.asarray((5, 5, 6, 6, 7, 8, 7, 8, 8)))
        t2 = hg.Tree(np.asarray((6, 6, 5, 5, 7, 7, 8, 8, 8)))
        t3 = hg.Tree(np.asarray((7, 7, 5, 5, 6, 6, 8, 8, 8)))

        self.assertTrue(hg.test_tree_isomorphism(t1, t2))
        self.assertTrue(hg.test_tree_isomorphism(t2, t1))
        self.assertTrue(hg.test_tree_isomorphism(t1, t3))
        self.assertTrue(hg.test_tree_isomorphism(t3, t1))
        self.assertTrue(hg.test_tree_isomorphism(t2, t3))
        self.assertTrue(hg.test_tree_isomorphism(t3, t2))

        t4 = hg.Tree(np.asarray((5, 5, 7, 6, 6, 8, 7, 8, 8)))

        self.assertTrue(not hg.test_tree_isomorphism(t1, t4))
        self.assertTrue(not hg.test_tree_isomorphism(t2, t4))
        self.assertTrue(not hg.test_tree_isomorphism(t3, t4))
        self.assertTrue(not hg.test_tree_isomorphism(t4, t1))
        self.assertTrue(not hg.test_tree_isomorphism(t4, t2))
        self.assertTrue(not hg.test_tree_isomorphism(t4, t3))

    def test_filter_non_relevant_node_from_tree(self):
        g = hg.get_4_adjacency_graph((1, 8))
        edge_weights = np.asarray((0, 2, 0, 0, 1, 0, 0))
        tree, altitudes = hg.bpt_canonical(g, edge_weights)

        def functor(tree, altitudes):
            area = hg.attribute_area(tree)
            area_min_children = hg.accumulate_parallel(tree, area, hg.Accumulators.min)
            return area_min_children < 3

        res_tree, res_altitudes = hg.filter_non_relevant_node_from_tree(tree, altitudes, functor)

        sm = hg.saliency(res_tree, res_altitudes)
        sm_ref = np.asarray((0, 0, 0, 0, 1, 0, 0))
        self.assertTrue(np.all(sm == sm_ref))

    def test_filter_small_node_from_tree(self):
        g = hg.get_4_adjacency_graph((1, 8))
        edge_weights = np.asarray((0, 2, 0, 0, 1, 0, 0))
        tree, altitudes = hg.quasi_flat_zone_hierarchy(g, edge_weights)

        res_tree, res_altitudes = hg.filter_small_nodes_from_tree(tree, altitudes, 3)

        sm = hg.saliency(res_tree, res_altitudes)
        sm_ref = np.asarray((0, 0, 0, 0, 1, 0, 0))
        self.assertTrue(np.all(sm == sm_ref))

    def test_filter_small_node_from_tree_on_rag(self):
        g = hg.get_4_adjacency_graph((2, 8))
        labels = np.asarray(((0, 1, 2, 3, 4, 5, 6, 7),
                             (0, 1, 2, 3, 4, 5, 6, 7)))

        rag = hg.make_region_adjacency_graph_from_labelisation(g, labels)
        edge_weights = np.asarray((0, 2, 0, 0, 1, 0, 0))

        tree, altitudes = hg.quasi_flat_zone_hierarchy(rag, edge_weights)

        res_tree, res_altitudes = hg.filter_small_nodes_from_tree(tree, altitudes, 5)

        sm = hg.saliency(res_tree, res_altitudes, handle_rag=False)
        sm_ref = np.asarray((0, 0, 0, 0, 1, 0, 0))
        self.assertTrue(np.all(sm == sm_ref))

        sm = hg.saliency(res_tree, res_altitudes, handle_rag=True)
        sm_ref = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0))
        self.assertTrue(np.all(sm == sm_ref))

    def test_filter_weak_frontier_nodes_from_tree(self):
        g = hg.get_4_adjacency_graph((1, 8))
        edge_weights = np.asarray((0, 2, 0, 0, 1, 0, 0))
        tree, altitudes = hg.bpt_canonical(g, edge_weights)

        res_tree, res_altitudes = hg.filter_weak_frontier_nodes_from_tree(tree, altitudes, edge_weights, 2)

        sm = hg.saliency(res_tree, res_altitudes)
        sm_ref = np.asarray((0, 2, 0, 0, 0, 0, 0))
        self.assertTrue(np.all(sm == sm_ref))

    def test_binary_labelisation_from_markers(self):
        tree = hg.Tree(np.asarray((9, 9, 9, 10, 10, 12, 13, 11, 11, 14, 12, 15, 13, 14, 15, 15)))
        object_marker = np.asarray((0, 1, 0, 1, 0, 0, 0, 0, 0), dtype=np.int8)
        background_marker = np.asarray((1, 0, 0, 0, 0, 0, 1, 0, 0), dtype=np.int8)

        labelisation = hg.binary_labelisation_from_markers(tree, object_marker, background_marker);

        ref_labelisation = np.asarray((0, 1, 0, 1, 1, 1, 0, 0, 0), dtype=np.int8)

        self.assertTrue(np.all(labelisation == ref_labelisation))

    def test_sort_hierarchy_with_altitudes(self):
        tree = hg.Tree(np.asarray((8, 8, 9, 9, 10, 10, 11, 13, 12, 12, 11, 13, 14, 14, 14)))
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 2, 4, 6, 5, 7))

        ntree, naltitudes, node_map = hg.sort_hierarchy_with_altitudes(tree, altitudes)

        ref_par = np.asarray((10, 10, 8, 8, 9, 9, 11, 12, 13, 11, 13, 12, 14, 14, 14))
        self.assertTrue(np.all(ref_par == ntree.parents()))

        ref_altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7))
        self.assertTrue(np.all(ref_altitudes == naltitudes))

        tree = hg.Tree(np.asarray((5, 5, 6, 6, 7, 7, 7, 7)))
        altitudes = np.asarray((0, 0, 1, 0, 0, 2, 1, 1))
        with self.assertRaises(ValueError):
            hg.sort_hierarchy_with_altitudes(tree, altitudes)

    def test_test_altitudes_increasingness(self):
        tree = hg.Tree(np.asarray((5, 5, 6, 6, 7, 7, 7, 7)))

        altitudes = np.asarray((0, 0, 0, 0, 0, 2, 1, 3))
        self.assertTrue(hg.test_altitudes_increasingness(tree, altitudes))

        altitudes = np.asarray((0, 0, 0, 0, 0, 1, 2, 2))
        self.assertTrue(hg.test_altitudes_increasingness(tree, altitudes))

        altitudes = np.asarray((2, 0, 0, 1, 2, 2, 1, 3))
        self.assertTrue(hg.test_altitudes_increasingness(tree, altitudes))

        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0))
        self.assertTrue(hg.test_altitudes_increasingness(tree, altitudes))

        altitudes = np.asarray((0, 0, 2, 0, 0, 2, 1, 3))
        self.assertFalse(hg.test_altitudes_increasingness(tree, altitudes))

        altitudes = np.asarray((0, 0, 1, 0, 0, 2, 1, 1))
        self.assertFalse(hg.test_altitudes_increasingness(tree, altitudes))


if __name__ == '__main__':
    unittest.main()
