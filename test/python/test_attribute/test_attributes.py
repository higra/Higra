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


class TestAttributes(unittest.TestCase):

    @staticmethod
    def get_test_tree():
        """
        base graph is

        (0)-- 0 --(1)-- 2 --(2)
         |         |         |
         6         6         0
         |         |         |
        (3)-- 0 --(4)-- 4 --(5)
         |         |         |
         5         5         3
         |         |         |
        (6)-- 0 --(7)-- 1 --(8)

        Minima are
        A: (0,1)
        B: (3,4)
        C: (2,5)
        D: (6,7)

        BPT:




        4                 +-------16------+
                          |               |
        3         +-------15-----+        |
                  |              |        |
        2     +---14--+          |        |
              |       |          |        |
        1     |       |       +--13-+     |
              |       |       |     |     |
        0   +-9-+   +-10+   +-12+   |   +-11+
            +   +   +   +   +   +   +   +   +
            0   1   2   5   6   7   8   3   4


        :return:
        """

        g = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0, 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1))

        return hg.bpt_canonical(g, edge_weights)

    def setUp(self):
        hg.clear_all_attributes()

    def test_area(self):
        tree, altitudes = TestAttributes.get_test_tree()

        ref_area = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 7, 9]
        area = hg.attribute_area(tree)
        self.assertTrue(np.allclose(ref_area, area))

        leaf_area = np.asarray([1, 2, 1, 1, 2, 1, 1, 1, 3])
        ref_area = [1, 2, 1, 1, 2, 1, 1, 1, 3, 3, 2, 3, 2, 5, 5, 10, 13]
        area = hg.attribute_area(tree, vertex_area=leaf_area)
        self.assertTrue(np.allclose(ref_area, area))

    def test_area_default_param(self):
        g = hg.get_4_adjacency_graph((2, 3))
        edge_weights = np.asarray((1, 4, 6, 5, 2, 7, 3))

        ref_area = (1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 6)

        tree, altitudes = hg.bpt_canonical(g, edge_weights)
        area = hg.attribute_area(tree)
        self.assertTrue(np.all(ref_area == area))

        tree2 = hg.Tree(tree.parents())
        area2 = hg.attribute_area(tree2)
        self.assertTrue(np.all(ref_area == area2))

    def test_attribute_volume(self):
        tree, altitudes = TestAttributes.get_test_tree()

        ref_attribute = [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 8, 2, 8, 12, 27, 35.]
        attribute = hg.attribute_volume(tree, altitudes)

        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_attribute_volume_with_area(self):
        tree = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7))
        altitudes = np.asarray((0, 0, 0, 0, 0, 2, 1, 4.))
        area = np.asarray((2, 1, 1, 3, 2, 3, 6, 9))

        ref_attribute = [0, 0, 0, 0, 0, 6, 18, 24]
        attribute = hg.attribute_volume(tree, altitudes, area)

        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_lca_map(self):
        tree, altitudes = TestAttributes.get_test_tree()

        ref_attribute = [9, 16, 14, 16, 10, 11, 16, 16, 16, 15, 12, 13]
        attribute = hg.attribute_lca_map(tree)
        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_frontier_length(self):
        tree, altitudes = TestAttributes.get_test_tree()

        ref_attribute = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 5]
        attribute = hg.attribute_frontier_length(tree)
        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_frontier_length_rag(self):
        g = hg.get_4_adjacency_graph((3, 3))
        vertex_labels = np.asarray(((0, 1, 1),
                                    (0, 2, 2),
                                    (3, 2, 4)))
        rag = hg.make_region_adjacency_graph_from_labelisation(g, vertex_labels)
        edge_weights = np.asarray((1, 5, 4, 3, 6, 2), dtype=np.float64)
        tree, altitudes = hg.bpt_canonical(rag, edge_weights)

        ref_attribute = np.asarray([0, 0, 0, 0, 0, 1, 2, 1, 4], dtype=np.float64)
        attribute = hg.attribute_frontier_length(tree)
        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_frontier_strength(self):
        tree, altitudes = TestAttributes.get_test_tree()
        edge_weights = np.asarray((0, 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1), dtype=np.float64)

        ref_attribute = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 26 / 5]

        attribute = hg.attribute_frontier_strength(tree, edge_weights)
        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_frontier_strength_rag(self):
        g = hg.get_4_adjacency_graph((3, 3))
        vertex_labels = np.asarray(((0, 1, 1),
                                    (0, 2, 2),
                                    (3, 2, 4)))
        rag = hg.make_region_adjacency_graph_from_labelisation(g, vertex_labels)
        rag_edge_weights = np.asarray((1, 5, 4, 3, 6, 2))
        tree, altitudes = hg.bpt_canonical(rag, rag_edge_weights)
        # tree is [5 5 6 7 6 7 8 8 8]
        edge_weights = np.asarray((1, 6, 2, 6, 1, 1, 5, 4, 5, 3, 1, 1), dtype=np.float64)

        ref_attribute = np.asarray([0, 0, 0, 0, 0, 1, 2, 5, 9 / 4], dtype=np.float64)
        attribute = hg.attribute_frontier_strength(tree, edge_weights)
        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_contour_length_partition_tree(self):
        tree, altitudes = TestAttributes.get_test_tree()
        ref_attribute = np.asarray([4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 8, 10, 16, 12], dtype=np.float64)
        attribute = hg.attribute_contour_length(tree)
        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_contour_length_partition_tree2(self):
        tree, altitudes = TestAttributes.get_test_tree()
        hg.set_attribute(hg.CptHierarchy.get_leaf_graph(tree), "no_border_vertex_out_degree", None)
        ref_attribute = np.asarray([2, 3, 2, 3, 4, 3, 2, 3, 2, 3, 3, 5, 3, 3, 4, 5, 0], dtype=np.float64)
        attribute = hg.attribute_contour_length(tree)
        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_contour_length_component_tree(self):
        g = hg.get_4_adjacency_graph((4, 4))

        # for reference, tree is a max tree on the following image
        # 0, 1, 4, 4,
        # 7, 5, 6, 8,
        # 2, 3, 4, 1,
        # 9, 8, 6, 7

        t = hg.Tree((28, 27, 24, 24,
                     20, 23, 22, 18,
                     26, 25, 24, 27,
                     16, 17, 21, 19,
                     17, 21, 22, 21, 23, 24, 23, 24, 25, 26, 27, 28, 28),
                    hg.TreeCategory.ComponentTree)

        res = hg.attribute_contour_length(t, leaf_graph=g)

        ref = np.asarray((4, 4, 4, 4,
                          4, 4, 4, 4,
                          4, 4, 4, 4,
                          4, 4, 4, 4,
                          4, 6, 4, 4, 4, 10, 6, 10, 22, 20, 18, 16, 16), dtype=np.float64)

        self.assertTrue(np.all(res == ref))

    def test_contour_length_tree_of_shapes(self):

        image = np.asarray(((0, 0, 0, 0),
                            (0, -2, 2, 0),
                            (0, -1, 1, 0),
                            (0, 0, 0, 0)))

        tree, altitudes = hg.component_tree_tree_of_shapes_image(image)

        res = hg.attribute_contour_length(tree)

        ref = np.asarray((4, 4, 4, 4,
                          4, 4, 4, 4,
                          4, 4, 4, 4,
                          4, 4, 4, 4,
                          4, 4, 6, 6, 16), dtype=np.float64)

        self.assertTrue(np.all(res == ref))

    def test_contour_length_rag_partition_tree(self):
        g = hg.get_4_adjacency_graph((3, 3))
        vertex_labels = np.asarray(((0, 1, 1),
                                    (0, 2, 2),
                                    (3, 2, 4)))
        rag = hg.make_region_adjacency_graph_from_labelisation(g, vertex_labels)
        edge_weights = np.asarray((1, 5, 4, 3, 6, 2), dtype=np.float64)
        tree, altitudes = hg.bpt_canonical(rag, edge_weights)

        ref_attribute = np.asarray([3, 3, 6, 2, 2, 4, 4, 4, 0], dtype=np.float64)
        attribute = hg.attribute_contour_length(tree)
        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_contour_strength_partition_tree(self):
        tree, altitudes = TestAttributes.get_test_tree()
        edge_weights = np.asarray((0, 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1), dtype=np.float64)
        attribute = hg.attribute_contour_strength(tree, edge_weights)

        ref_perimeter = np.asarray([4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 8, 10, 16, 12], dtype=np.float64)
        ref_weights = np.asarray([6, 8, 2, 11, 15, 7, 5, 6, 4, 14, 9, 26, 11, 13, 19, 26, 0], dtype=np.float64)

        self.assertTrue(np.allclose(ref_weights / ref_perimeter, attribute))

    def test_contour_strength_partition_tree2(self):
        tree, altitudes = TestAttributes.get_test_tree()
        edge_weights = np.asarray((0, 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1), dtype=np.float64)
        hg.set_attribute(hg.CptHierarchy.get_leaf_graph(tree), "no_border_vertex_out_degree", None)
        attribute = hg.attribute_contour_strength(tree, edge_weights)

        ref_perimeter = np.asarray([2, 3, 2, 3, 4, 3, 2, 3, 2, 3, 3, 5, 3, 3, 4, 5, 1], dtype=np.float64)
        ref_weights = np.asarray([6, 8, 2, 11, 15, 7, 5, 6, 4, 14, 9, 26, 11, 13, 19, 26, 0], dtype=np.float64)
        self.assertTrue(np.allclose(ref_weights / ref_perimeter, attribute))

    def test_contour_strength_component_tree(self):
        g = hg.get_4_adjacency_graph((4, 4))

        edge_weights = np.arange(g.num_edges(), dtype=np.float64)
        # for reference, tree is a max tree on the following image
        # 0, 1, 4, 4,
        # 7, 5, 6, 8,
        # 2, 3, 4, 1,
        # 9, 8, 6, 7

        t = hg.Tree((28, 27, 24, 24,
                     20, 23, 22, 18,
                     26, 25, 24, 27,
                     16, 17, 21, 19,
                     17, 21, 22, 21, 23, 24, 23, 24, 25, 26, 27, 28, 28),
                    hg.TreeCategory.ComponentTree)
        hg.CptHierarchy.link(t, g)
        res = hg.attribute_contour_strength(t, edge_weights)

        ref_perimeter = np.asarray((4, 4, 4, 4,
                                    4, 4, 4, 4,
                                    4, 4, 4, 4,
                                    4, 4, 4, 4,
                                    4, 6, 4, 4, 4, 10, 6, 10, 22, 20, 18, 16, 16), dtype=np.float64)

        ref_weights = np.asarray((1, 5, 11, 10,
                                  16, 29, 37, 30,
                                  37, 57, 65, 51,
                                  36, 60, 64, 43,
                                  36, 54, 30, 43, 16, 71, 45, 58, 123, 94, 57, 1, 0), dtype=np.float64)

        self.assertTrue(np.allclose(ref_weights / ref_perimeter, res))

    def test_contour_strength_rag_partition_tree(self):
        g = hg.get_4_adjacency_graph((3, 3))
        vertex_labels = np.asarray(((0, 1, 1),
                                    (0, 2, 2),
                                    (3, 2, 4)))

        base_edge_weights = np.arange(g.num_edges(), dtype=np.float64)

        rag = hg.make_region_adjacency_graph_from_labelisation(g, vertex_labels)
        edge_weights = np.asarray((1, 5, 4, 3, 6, 2))
        tree, altitudes = hg.bpt_canonical(rag, edge_weights)

        attribute = hg.attribute_contour_strength(tree, base_edge_weights)

        ref_perimeter = np.asarray([3, 3, 6, 2, 2, 4, 4, 4, 1], dtype=np.float64)
        ref_weights = np.asarray([11, 7, 42, 16, 20, 18, 22, 22, 0], dtype=np.float64)

        self.assertTrue(np.allclose(ref_weights / ref_perimeter, attribute))

    def test_compactness(self):
        tree, altitudes = TestAttributes.get_test_tree()
        ref_attribute = [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.88888889, 0.88888889, 0.88888889, 0.88888889, 0.75, 0.64,
                         0.4375, 1.]
        attribute = hg.attribute_compactness(tree)
        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_mean_vertex_weights(self):
        tree, altitudes = TestAttributes.get_test_tree()

        leaf_data = np.asarray((0, 1, 2, 3, 4, 5, 6, 7, 8),
                               dtype=np.float64)
        ref_attribute = np.asarray((0, 1, 2, 3, 4, 5, 6, 7, 8,
                                    1. / 2, 7. / 2, 7. / 2, 13. / 2, 7.,
                                    2., 29. / 7, 4.))

        attribute = hg.attribute_mean_vertex_weights(tree, vertex_weights=leaf_data)
        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_mean_vertex_weights_vectorial(self):
        tree, altitudes = TestAttributes.get_test_tree()

        leaf_data = np.asarray(((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)),
                               dtype=np.float64)
        ref_attribute = np.asarray(((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
                                    (1. / 2, 1. / 2), (7. / 2, 7. / 2), (7. / 2, 7. / 2), (13. / 2, 13. / 2), (7., 7.),
                                    (2., 2.), (29. / 7, 29. / 7), (4., 4.)))

        attribute = hg.attribute_mean_vertex_weights(tree, vertex_weights=leaf_data)
        self.assertTrue(np.allclose(ref_attribute, attribute))

    def test_sibling(self):
        t = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7))
        ref = np.asarray((1, 0, 3, 4, 2, 6, 5, 7))
        res = hg.attribute_sibling(t)
        self.assertTrue(np.all(ref == res))

        t = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7))
        ref2 = np.asarray((1, 0, 4, 2, 3, 6, 5, 7))
        res2 = hg.attribute_sibling(t, -1)
        self.assertTrue(np.all(ref2 == res2))

    def test_depth(self):
        t = hg.Tree((6, 6, 7, 8, 8, 8, 7, 9, 9, 9))
        ref = np.asarray((3, 3, 2, 2, 2, 2, 2, 1, 1, 0))
        res = hg.attribute_depth(t)
        self.assertTrue(np.all(ref == res))

    def test_regular_altitudes(self):
        t = hg.Tree((6, 6, 7, 8, 8, 8, 7, 9, 9, 9))
        ref = np.asarray((0, 0, 0, 0, 0, 0, 1 / 3, 2 / 3, 2 / 3, 1))
        res = hg.attribute_regular_altitudes(t)
        self.assertTrue(np.allclose(ref, res))

    def test_vertex_coordinates(self):
        g = hg.get_4_adjacency_graph((2, 3))
        ref = np.asarray((((0, 0), (0, 1), (0, 2)),
                          ((1, 0), (1, 1), (1, 2))))
        res = hg.attribute_vertex_coordinates(g)
        self.assertTrue(np.allclose(ref, res))

    def test_edge_length(self):
        g = hg.get_4_adjacency_graph((2, 3))
        ref = np.asarray((1, 1, 1, 1, 1, 1, 1))
        res = hg.attribute_edge_length(g)
        self.assertTrue(np.allclose(ref, res))

    def test_edge_length_rag(self):
        g = hg.get_4_adjacency_graph((2, 3))
        vertex_labels = np.asarray(((1, 2, 2),
                                    (3, 3, 3)))
        rag = hg.make_region_adjacency_graph_from_labelisation(g, vertex_labels)
        ref = np.asarray((1, 1, 2))
        res = hg.attribute_edge_length(rag)
        self.assertTrue(np.allclose(ref, res))

    def test_vertex_perimeter(self):
        g = hg.get_4_adjacency_graph((2, 3))
        ref = np.asarray(((4, 4, 4),
                          (4, 4, 4)))
        res = hg.attribute_vertex_perimeter(g)
        self.assertTrue(np.allclose(ref, res))

    def test_vertex_perimeter2(self):
        g = hg.get_4_adjacency_graph((2, 3))
        hg.set_attribute(g, "no_border_vertex_out_degree", None)
        ref = np.asarray(((2, 3, 2),
                          (2, 3, 2)))
        res = hg.attribute_vertex_perimeter(g)
        self.assertTrue(np.allclose(ref, res))

    def test_vertex_perimeter_rag(self):
        g = hg.get_4_adjacency_graph((2, 3))
        vertex_labels = np.asarray(((1, 2, 2),
                                    (3, 3, 3)))
        rag = hg.make_region_adjacency_graph_from_labelisation(g, vertex_labels)
        ref = np.asarray((2, 3, 3))
        res = hg.attribute_vertex_perimeter(rag)
        self.assertTrue(np.allclose(ref, res))

    def test_attribute_vertex_list(self):
        tree, altitudes = TestAttributes.get_test_tree()

        res = hg.attribute_vertex_list(tree)
        ref = [[0], [1], [2], [3], [4], [5], [6], [7], [8],
               [0, 1], [2, 5], [3, 4], [6, 7], [6, 7, 8],
               [0, 1, 2, 5], [0, 1, 2, 5, 6, 7, 8],
               [0, 1, 2, 5, 6, 7, 8, 3, 4]]
        self.assertTrue(len(ref) == len(res))
        for i in range(len(ref)):
            self.assertTrue(set(ref[i]) == set(res[i]))

    def test_attribute_gaussian_region_weights_model_scalar(self):
        tree, altitudes = TestAttributes.get_test_tree()
        vertex_list = hg.attribute_vertex_list(tree)

        np.random.seed(42)
        vertex_weights = np.random.rand(tree.num_leaves())
        mean, variance = hg.attribute_gaussian_region_weights_model(tree, vertex_weights)

        for i in tree.leaves_to_root_iterator():
            m = np.mean(vertex_weights[vertex_list[i]])
            v = np.var(vertex_weights[vertex_list[i]])
            self.assertTrue(np.isclose(m, mean[i]))
            self.assertTrue(np.isclose(v, variance[i]))

    def test_attribute_gaussian_region_weights_model_scalar_int(self):
        tree, altitudes = TestAttributes.get_test_tree()
        vertex_list = hg.attribute_vertex_list(tree)

        np.random.seed(42)
        vertex_weights = np.random.randint(255, size=tree.num_leaves())
        mean, variance = hg.attribute_gaussian_region_weights_model(tree, vertex_weights)

        for i in tree.leaves_to_root_iterator():
            m = np.mean(vertex_weights[vertex_list[i]])
            v = np.var(vertex_weights[vertex_list[i]])
            self.assertTrue(np.isclose(m, mean[i]))
            self.assertTrue(np.isclose(v, variance[i]))

    def test_attribute_gaussian_region_weights_model_vectorial(self):
        tree, altitudes = TestAttributes.get_test_tree()
        vertex_list = hg.attribute_vertex_list(tree)

        np.random.seed(42)
        vertex_weights = np.random.rand(tree.num_leaves(), 3)
        mean, variance = hg.attribute_gaussian_region_weights_model(tree, vertex_weights)

        for i in tree.leaves_to_root_iterator(include_leaves=True):
            m = np.mean(vertex_weights[vertex_list[i]], 0)

            self.assertTrue(np.allclose(m, mean[i, :]))

            # numpy wrongly interprets a single observation with several variables as
            # multiple observations of a single variables
            if i >= tree.num_leaves():
                v = np.cov(vertex_weights[vertex_list[i]], rowvar=False, bias=True)
                self.assertTrue(np.allclose(v, variance[i, ...]))
            else:
                v = np.zeros_like(variance[i, ...])
                self.assertTrue(np.allclose(v, variance[i, ...]))

    def test_tree_attribute_extrema(self):
        t = hg.Tree((11, 11, 9, 9, 8, 8, 13, 13, 10, 10, 12, 12, 14, 14, 14))

        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 4, 8, 10.))
        ref = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0))
        res = hg.attribute_extrema(t, altitudes)
        self.assertTrue(np.all(ref == res))

    def test_tree_attribute_extrema2(self):
        graph = hg.get_4_adjacency_implicit_graph((4, 4))
        vertex_weights = np.asarray((0, 1, 4, 4,
                                     7, 5, 6, 8,
                                     2, 3, 4, 1,
                                     9, 8, 6, 7))

        tree, altitudes = hg.component_tree_max_tree(graph, vertex_weights)

        extrema = hg.attribute_extrema(tree, altitudes)
        expected_extrema = np.asarray((0, 0, 0, 0,
                                       0, 0, 0, 0,
                                       0, 0, 0, 0,
                                       0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0))
        self.assertTrue(np.all(expected_extrema == extrema))

    def test_attribute_extinction_value(self):
        # same as dynamics
        t = hg.Tree((8, 8, 9, 7, 7, 11, 11, 9, 10, 10, 12, 12, 12))
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 10.))
        attribute = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 4, 2, 10.))
        ref = np.asarray((3, 3, 0, 10, 10, 2, 2, 10, 3, 10, 10, 2, 10))
        res = hg.attribute_extinction_value(t, altitudes, attribute)
        self.assertTrue(np.all(ref == res))

        res = hg.attribute_extinction_value(t, altitudes, attribute, True)
        self.assertTrue(np.all(ref == res))

        res = hg.attribute_extinction_value(t, altitudes, attribute, "increasing")
        self.assertTrue(np.all(ref == res))

    def test_attribute_extinction_value2(self):
        graph = hg.get_4_adjacency_implicit_graph((4, 4))
        vertex_weights = np.asarray((0, 1, 4, 4,
                                     7, 5, 6, 8,
                                     2, 3, 4, 1,
                                     9, 8, 6, 7))

        tree, altitudes = hg.component_tree_max_tree(graph, vertex_weights)
        area = hg.attribute_area(tree)

        expected_ext = np.asarray((0, 0, 0, 0,
                                   1, 0, 0, 4,
                                   0, 0, 0, 0,
                                   16, 0, 0, 1,
                                   16, 16, 4, 1, 1, 16, 4, 4, 16, 16, 16, 16, 16))

        ext = hg.attribute_extinction_value(tree, altitudes, area)
        self.assertTrue(np.all(expected_ext == ext))

        ext = hg.attribute_extinction_value(tree, altitudes, area, False)
        self.assertTrue(np.all(expected_ext == ext))

        ext = hg.attribute_extinction_value(tree, altitudes, area, "decreasing")
        self.assertTrue(np.all(expected_ext == ext))

    def test_attribute_height_inc(self):
        t = hg.Tree((7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11))
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 3, 2, 1, 5, 8.))
        ref = np.asarray((0, 0, 0, 0, 0, 0, 0, 2, 3, 7, 6, 7))

        res = hg.attribute_height(t, altitudes)
        self.assertTrue(np.all(res == ref))

        res = hg.attribute_height(t, altitudes, True)
        self.assertTrue(np.all(res == ref))

        res = hg.attribute_height(t, altitudes, "increasing")
        self.assertTrue(np.all(res == ref))

    def test_attribute_height_dec(self):
        t = hg.Tree((7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11))
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 8, 5, 9, 4, 1.))
        ref = np.asarray((0, 0, 0, 0, 0, 0, 0, 4, 1, 8, 7, 8))

        res = hg.attribute_height(t, altitudes)
        self.assertTrue(np.all(res == ref))

        res = hg.attribute_height(t, altitudes, False)
        self.assertTrue(np.all(res == ref))

        res = hg.attribute_height(t, altitudes, "decreasing")
        self.assertTrue(np.all(res == ref))

    def test_attribute_height_current_inc(self):
        graph = hg.get_4_adjacency_implicit_graph((10, 10))
        vertex_weights = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 4, 3, 4, 0, 0, 0, 2, 2, 0],
                                   [0, 2, 3, 3, 2, 1, 2, 2, 0, 0],
                                   [0, 2, 4, 3, 2, 1, 0, 1, 0, 0],
                                   [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 3, 0, 4, 2, 0, 0],
                                   [0, 1, 1, 0, 3, 2, 4, 4, 0, 0],
                                   [0, 1, 1, 0, 3, 0, 4, 4, 1, 0],
                                   [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        ref_map = np.array([[4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 0, 1, 0, 4, 4, 4, 0, 0, 4],
                            [4, 2, 1, 1, 2, 3, 0, 0, 4, 4],
                            [4, 2, 0, 1, 2, 3, 4, 3, 4, 4],
                            [4, 4, 2, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 0, 4, 0, 2, 4, 4],
                            [4, 0, 0, 4, 0, 2, 0, 0, 4, 4],
                            [4, 0, 0, 4, 0, 4, 0, 0, 3, 4],
                            [4, 0, 0, 4, 4, 4, 4, 3, 3, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]], dtype=np.uint8)

        tree, altitudes = hg.component_tree_max_tree(graph, vertex_weights)
        height = hg.attribute_height(tree, altitudes, True, "current")
        map = hg.reconstruct_leaf_data(tree, height)

        self.assertTrue(np.allclose(ref_map, map))

    def test_attribute_height_current_dec(self):
        graph = hg.get_4_adjacency_implicit_graph((10, 10))
        vertex_weights = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 4, 3, 4, 0, 0, 0, 2, 2, 0],
                                   [0, 2, 3, 3, 2, 1, 2, 2, 0, 0],
                                   [0, 2, 4, 3, 2, 1, 0, 1, 0, 0],
                                   [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 3, 0, 4, 2, 0, 0],
                                   [0, 1, 1, 0, 3, 2, 4, 4, 0, 0],
                                   [0, 1, 1, 0, 3, 0, 4, 4, 1, 0],
                                   [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        ref_map = np.array([[4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [4, 0, 1, 0, 4, 4, 4, 0, 0, 4],
                            [4, 2, 1, 1, 2, 3, 0, 0, 4, 4],
                            [4, 2, 0, 1, 2, 3, 4, 3, 4, 4],
                            [4, 4, 2, 4, 4, 4, 4, 4, 4, 4],
                            [4, 4, 4, 4, 0, 4, 0, 2, 4, 4],
                            [4, 0, 0, 4, 0, 2, 0, 0, 4, 4],
                            [4, 0, 0, 4, 0, 4, 0, 0, 3, 4],
                            [4, 0, 0, 4, 4, 4, 4, 3, 3, 4],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]], dtype=np.uint8)

        tree, altitudes = hg.component_tree_min_tree(graph, vertex_weights.max()-vertex_weights)
        height = hg.attribute_height(tree, altitudes, False, "current")
        map = hg.reconstruct_leaf_data(tree, height)

        self.assertTrue(np.allclose(ref_map, map))

    def test_attribute_dynamics(self):
        t = hg.Tree((8, 8, 9, 7, 7, 11, 11, 9, 10, 10, 12, 12, 12))
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 10.))
        ref = np.asarray((3, 3, 0, 10, 10, 2, 2, 10, 3, 10, 10, 2, 10))

        res = hg.attribute_dynamics(t, altitudes)
        self.assertTrue(np.all(res == ref))

    def test_attribute_dynamics2(self):
        t = hg.Tree((11, 11, 9, 9, 8, 8, 13, 13, 10, 10, 12, 12, 14, 14, 14))
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 4, 8, 10.))
        ref = np.asarray((3, 3, 0, 0, 10, 10, 2, 2, 10, 0, 10, 3, 10, 2, 10))

        res = hg.attribute_dynamics(t, altitudes)
        self.assertTrue(np.all(res == ref))

    def test_attribute_child_number(self):
        tree = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7))

        ref = np.asarray((0, 1, 0, 1, 2, 0, 1, -1), dtype=np.int64)

        res = hg.attribute_child_number(tree)
        self.assertTrue(np.all(res == ref))

    def test_attribute_children_pair_sum_product_scalar(self):
        tree = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7))

        node_weights = np.asarray((2, 3, 4, 5, 6, 7, 8, 9))
        res = hg.attribute_children_pair_sum_product(tree, node_weights)
        ref = np.asarray((0, 0, 0, 0, 0, 6, 74, 56))
        self.assertTrue(np.allclose(ref, res))

    def test_attribute_children_pair_sum_product_scalar(self):
        tree = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7))

        node_weights = np.asarray((2, 9, 3, 8, 4, 7, 5, 6, 6, 5, 7, 4, 8, 3, 9, 2)).reshape((8, 2))
        res = hg.attribute_children_pair_sum_product(tree, node_weights)
        ref = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 72, 74, 107, 56, 12)).reshape((8, 2))
        self.assertTrue(np.allclose(ref, res))

    def test_attribute_tree_sampling_probability_edge_model(self):
        g = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0, 6, 2, 6, 0, 0, 5, 4, 5, 3, 2, 2))
        tree, altitudes = hg.quasi_flat_zone_hierarchy(g, edge_weights)
        res = hg.attribute_tree_sampling_probability(tree, g, edge_weights, model='edge')

        ref = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 3, 26)) / np.sum(edge_weights)
        self.assertTrue(np.allclose(ref, res))

    def test_attribute_tree_sampling_probability_null_model(self):
        g = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0, 6, 2, 6, 0, 0, 5, 4, 5, 3, 2, 2))
        tree, altitudes = hg.quasi_flat_zone_hierarchy(g, edge_weights)
        res = hg.attribute_tree_sampling_probability(tree, g, edge_weights, model='null')

        Z = np.sum(edge_weights)
        ref = np.asarray(
            (0, 0, 0, 0, 0, 0, 0, 0, 0,
             6 * 8, 2 * 7,
             11 * 15,
             6 * 2 + 6 * 7 + 8 * 2 + 8 * 7,
             7 * 9 + 7 * 5 + 9 * 5,
             6 * 7 + 6 * 9 + 6 * 5 + 8 * 7 + 8 * 9 + 8 * 5 + 2 * 7 + 2 * 9 + 2 * 5 + 7 * 7 + 7 * 9 + 7 * 5,
             6 * 11 + 6 * 15 + 8 * 11 + 8 * 15 + 2 * 11 + 2 * 15 + 11 * 7 + 11 * 7 + 11 * 9 + 11 * 5 + 15 * 7 + 15 * 7 + 15 * 9 + 15 * 5)) / \
              (Z * Z)
        self.assertTrue(np.allclose(ref, res))

    def test_topological_height(self):
        tree = hg.Tree((6, 6, 9, 9, 7, 8, 10, 8, 10, 10, 10))
        res = hg.attribute_topological_height(tree)
        ref = (0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 3)
        self.assertTrue(np.array_equal(res, ref))
        
    def test_moment_of_inertia(self):
        """
        The test image is a binary image of 5x15 pixels:
        
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 1 0 0 1 0 0 0 0 0 1 1 1 0
        0 1 1 1 0 1 0 1 1 1 0 1 1 1 0
        0 0 1 0 0 1 0 0 0 0 0 1 1 1 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        
        The max-tree of this image is composed of five non-leaf nodes: 
        the root and the four maxima (a cross, a vertical line, a horizontal line and a square)
        Using the formula given in https://en.wikipedia.org/wiki/Image_moment, the moment of inertia of each shape is:
          - Cross: 0.16
          - Square: 0.1481
          - Vertical and horizontal lines: 0.2222
          - Rectangle (root): 0.2756
        The moment of inertia of the leaf nodes are set to 0.0
        
       
        """
        graph = hg.get_4_adjacency_implicit_graph((5, 15))
        vertex_weights = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                                     [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
                                     [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        tree, altitudes = hg.component_tree_max_tree(graph, vertex_weights)
        res = hg.attribute_moment_of_inertia(tree)        
        ref = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0.1481, 0.2222, 0.16, 0.2222, 0.2756)
        self.assertTrue(np.allclose(res,ref,atol=0.0001))


if __name__ == '__main__':
    unittest.main()
