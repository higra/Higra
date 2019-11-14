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


class TestTreeOfShapesImage(unittest.TestCase):

    def test_tree_of_shapes_no_padding(self):
        image = np.asarray(((1, 1, 1, 1, 1, 1),
                            (1, 0, 0, 3, 3, 1),
                            (1, 0, 1, 1, 3, 1),
                            (1, 0, 0, 3, 3, 1),
                            (1, 1, 1, 1, 1, 1)), dtype=np.int8)

        tree, altitudes = hg.component_tree_tree_of_shapes_image2d(image, 'none', False)
        ref_parents = np.asarray((101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
                                  101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
                                  101, 101, 100, 100, 100, 101, 99, 99, 99, 101, 101,
                                  101, 101, 100, 101, 101, 101, 101, 101, 99, 101, 101,
                                  101, 101, 100, 101, 101, 101, 101, 101, 99, 101, 101,
                                  101, 101, 100, 101, 101, 101, 101, 101, 99, 101, 101,
                                  101, 101, 100, 100, 100, 101, 99, 99, 99, 101, 101,
                                  101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
                                  101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
                                  101, 101, 101), dtype=np.int64)

        ref_altitudes = np.asarray((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 0, 0, 0, 1, 3, 3, 3, 1, 1,
                                    1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 1,
                                    1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 1,
                                    1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 1,
                                    1, 1, 0, 0, 0, 1, 3, 3, 3, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    3, 0, 1), dtype=np.int8)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(altitudes == ref_altitudes))

        leaf_graph = hg.CptHierarchy.get_leaf_graph(tree)
        res_shape = hg.CptGridGraph.get_shape(leaf_graph)
        self.assertTrue(len(res_shape) == 2)
        self.assertTrue(res_shape[0] * res_shape[1] == tree.num_leaves())
        self.assertTrue(res_shape[0] == image.shape[0] * 2 - 1)
        self.assertTrue(res_shape[1] == image.shape[1] * 2 - 1)

    def test_tree_of_shapes_no_padding_original_space(self):
        image = np.asarray(((1, 1, 1, 1, 1, 1),
                            (1, 0, 0, 3, 3, 1),
                            (1, 0, 1, 1, 3, 1),
                            (1, 0, 0, 3, 3, 1),
                            (1, 1, 1, 1, 1, 1)), dtype=np.float64)

        tree, altitudes = hg.component_tree_tree_of_shapes_image2d(image, 'none', True)
        ref_parents = np.asarray((32, 32, 32, 32, 32, 32,
                                  32, 30, 30, 31, 31, 32,
                                  32, 30, 32, 32, 31, 32,
                                  32, 30, 30, 31, 31, 32,
                                  32, 32, 32, 32, 32, 32,
                                  32, 32, 32), dtype=np.int64)

        ref_altitudes = np.asarray((1, 1, 1, 1, 1, 1,
                                    1, 0, 0, 3, 3, 1,
                                    1, 0, 1, 1, 3, 1,
                                    1, 0, 0, 3, 3, 1,
                                    1, 1, 1, 1, 1, 1,
                                    0, 3, 1), dtype=np.float64)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(altitudes == ref_altitudes))

        leaf_graph = hg.CptHierarchy.get_leaf_graph(tree)
        res_shape = hg.CptGridGraph.get_shape(leaf_graph)
        self.assertTrue(len(res_shape) == 2)
        self.assertTrue(res_shape[0] * res_shape[1] == tree.num_leaves())
        self.assertTrue(res_shape[0] == image.shape[0])
        self.assertTrue(res_shape[1] == image.shape[1])

    def test_tree_of_shapes_padding_0(self):
        image = np.asarray(((1, 1, 1),
                            (1, -2, 3)), dtype=np.int32)

        tree, altitudes = hg.component_tree_tree_of_shapes_image2d(image, 'zero', False)
        ref_parents = np.asarray((66, 66, 66, 66, 66, 66, 66, 66, 66,
                                  66, 66, 66, 66, 66, 66, 66, 66, 66,
                                  66, 66, 65, 65, 65, 65, 65, 66, 66,
                                  66, 66, 65, 66, 66, 66, 65, 66, 66,
                                  66, 66, 65, 66, 63, 66, 64, 66, 66,
                                  66, 66, 66, 66, 66, 66, 66, 66, 66,
                                  66, 66, 66, 66, 66, 66, 66, 66, 66,
                                  66, 65, 66, 66), dtype=np.int64)

        ref_altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 1, 1, 1, 1, 1, 0, 0,
                                    0, 0, 1, 0, 0, 0, 1, 0, 0,
                                    0, 0, 1, 0, -2, 0, 3, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    -2, 3, 1, 0), dtype=np.int32)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(altitudes == ref_altitudes))

        leaf_graph = hg.CptHierarchy.get_leaf_graph(tree)
        res_shape = hg.CptGridGraph.get_shape(leaf_graph)
        self.assertTrue(len(res_shape) == 2)
        self.assertTrue(res_shape[0] * res_shape[1] == tree.num_leaves())
        self.assertTrue(res_shape[0] == (image.shape[0] + 2) * 2 - 1)
        self.assertTrue(res_shape[1] == (image.shape[1] + 2) * 2 - 1)

    def test_tree_of_shapes_padding_0_original_space(self):
        image = np.asarray(((1, 1, 1),
                            (1, -2, 3)), dtype=np.int32)

        tree, altitudes = hg.component_tree_tree_of_shapes_image2d(image, 'zero', True)
        ref_parents = np.asarray((7, 7, 7,
                                  7, 8, 6,
                                  7, 9, 9, 9), dtype=np.int64)

        ref_altitudes = np.asarray((1, 1, 1,
                                    1, -2, 3,
                                    3, 1, -2, 0), dtype=np.int32)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(altitudes == ref_altitudes))

        leaf_graph = hg.CptHierarchy.get_leaf_graph(tree)
        res_shape = hg.CptGridGraph.get_shape(leaf_graph)
        self.assertTrue(len(res_shape) == 2)
        self.assertTrue(res_shape[0] * res_shape[1] == tree.num_leaves())
        self.assertTrue(res_shape[0] == image.shape[0])
        self.assertTrue(res_shape[1] == image.shape[1])

    def test_tree_of_shapes_padding_mean_original_space(self):
        image = np.asarray(((1, 1),
                            (1, -2),
                            (1, 7)), dtype=np.float64)

        tree, altitudes = hg.component_tree_tree_of_shapes_image2d(image, 'mean', True)
        ref_parents = np.asarray((7, 7,
                                  7, 6,
                                  7, 8,
                                  7, 9, 9, 9), dtype=np.int64)

        ref_altitudes = np.asarray((1., 1.,
                                    1., -2.,
                                    1., 7.,
                                    -2., 1., 7., 1.5), dtype=np.float64)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.allclose(altitudes, ref_altitudes))

    def test_tree_of_shapes_no_immersion_no_padding_original_space(self):
        image = np.asarray(((1, 1, 1, 1, 1),
                            (1, 0, 1, 2, 1),
                            (1, 1, 1, 1, 1)), dtype=np.float64)

        tree, altitudes = hg.component_tree_tree_of_shapes_image2d(image, 'none', original_size=True, immersion=False)
        ref_parents = np.asarray((17, 17, 17, 17, 17,
                                  17, 16, 17, 15, 17,
                                  17, 17, 17, 17, 17, 17, 17, 17), dtype=np.int64)

        ref_altitudes = np.asarray((1, 1, 1, 1, 1,
                                    1, 0, 1, 2, 1,
                                    1, 1, 1, 1, 1, 2, 0, 1), dtype=np.float64)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.allclose(altitudes, ref_altitudes))

        g = hg.CptHierarchy.get_leaf_graph(tree)
        s = hg.CptGridGraph.get_shape(g)
        self.assertTrue(s[0] * s[1] == tree.num_leaves())

    def test_tree_of_shapes_no_immersion_padding_zero_original_space(self):
        image = np.asarray(((1, 1, 1, 1, 1),
                            (1, 0, 1, 2, 1),
                            (1, 1, 1, 1, 1)), dtype=np.float64)

        tree, altitudes = hg.component_tree_tree_of_shapes_image2d(image, 'zero', original_size=True, immersion=False)
        ref_parents = np.asarray((17, 17, 17, 17, 17,
                                  17, 15, 17, 16, 17,
                                  17, 17, 17, 17, 17, 17, 17, 18, 18), dtype=np.int64)

        ref_altitudes = np.asarray((1, 1, 1, 1, 1,
                                    1, 0, 1, 2, 1,
                                    1, 1, 1, 1, 1, 0, 2, 1, 0), dtype=np.float64)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.allclose(altitudes, ref_altitudes))

        g = hg.CptHierarchy.get_leaf_graph(tree)
        s = hg.CptGridGraph.get_shape(g)
        self.assertTrue(s[0] * s[1] == tree.num_leaves())

    def test_tree_of_shapes_no_immersion_no_padding_no_original_space(self):
        image = np.asarray(((1, 1, 1, 1, 1),
                            (1, 0, 1, 2, 1),
                            (1, 1, 1, 1, 1)), dtype=np.float64)

        tree, altitudes = hg.component_tree_tree_of_shapes_image2d(image, 'none', original_size=False, immersion=False)
        ref_parents = np.asarray((17, 17, 17, 17, 17,
                                  17, 16, 17, 15, 17,
                                  17, 17, 17, 17, 17, 17, 17, 17), dtype=np.int64)

        ref_altitudes = np.asarray((1, 1, 1, 1, 1,
                                    1, 0, 1, 2, 1,
                                    1, 1, 1, 1, 1, 2, 0, 1), dtype=np.float64)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.allclose(altitudes, ref_altitudes))

        g = hg.CptHierarchy.get_leaf_graph(tree)
        s = hg.CptGridGraph.get_shape(g)
        self.assertTrue(s[0] * s[1] == tree.num_leaves())

    def test_tree_of_shapes_no_immersion_padding_zero_no_original_space(self):
        image = np.asarray(((1, 1, 1, 1, 1),
                            (1, 0, 1, 2, 1),
                            (1, 1, 1, 1, 1)), dtype=np.float64)

        tree, altitudes = hg.component_tree_tree_of_shapes_image2d(image, 'zero', original_size=False, immersion=False)
        ref_parents = np.asarray((38, 38, 38, 38, 38, 38, 38,
                                  38, 37, 37, 37, 37, 37, 38,
                                  38, 37, 36, 37, 35, 37, 38,
                                  38, 37, 37, 37, 37, 37, 38,
                                  38, 38, 38, 38, 38, 38, 38, 37, 37, 38, 38), dtype=np.int64)

        ref_altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0,
                                    0, 1, 1, 1, 1, 1, 0,
                                    0, 1, 0, 1, 2, 1, 0,
                                    0, 1, 1, 1, 1, 1, 0,
                                    0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0), dtype=np.float64)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.allclose(altitudes, ref_altitudes))

        g = hg.CptHierarchy.get_leaf_graph(tree)
        s = hg.CptGridGraph.get_shape(g)
        self.assertTrue(s[0] * s[1] == tree.num_leaves())

    def test_tree_of_shapes_self_dual(self):
        np.random.seed(42)
        image = np.random.rand(25, 38)
        neg_image = -1 * image

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image2d(image)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image2d(neg_image)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))
