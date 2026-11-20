############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import inspect
import unittest
from unittest import mock
import higra as hg
import higra.image.tree_of_shapes as tos_module
import numpy as np


class TestTreeOfShapesImage(unittest.TestCase):

    def test_tree_of_shapes_2d_no_padding(self):
        image = np.asarray(((1, 1, 1, 1, 1, 1),
                            (1, 0, 0, 3, 3, 1),
                            (1, 0, 1, 1, 3, 1),
                            (1, 0, 0, 3, 3, 1),
                            (1, 1, 1, 1, 1, 1)), dtype=np.int8)

        tree, altitudes = hg.component_tree_tree_of_shapes_image(image, 'none', False)
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

    def test_tree_of_shapes_2d_no_padding_original_space(self):
        image = np.asarray(((1, 1, 1, 1, 1, 1),
                            (1, 0, 0, 3, 3, 1),
                            (1, 0, 1, 1, 3, 1),
                            (1, 0, 0, 3, 3, 1),
                            (1, 1, 1, 1, 1, 1)), dtype=np.float64)

        tree, altitudes = hg.component_tree_tree_of_shapes_image(image, 'none', True)
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

    def test_tree_of_shapes_2d_padding_0(self):
        image = np.asarray(((1, 1, 1),
                            (1, -2, 3)), dtype=np.int32)

        tree, altitudes = hg.component_tree_tree_of_shapes_image(image, 'zero', False)
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

    def test_tree_of_shapes_2d_padding_0_original_space(self):
        image = np.asarray(((1, 1, 1),
                            (1, -2, 3)), dtype=np.int32)

        tree, altitudes = hg.component_tree_tree_of_shapes_image(image, 'zero', True)
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

    def test_tree_of_shapes_2d_padding_mean_original_space(self):
        image = np.asarray(((1, 1),
                            (1, -2),
                            (1, 7)), dtype=np.float64)

        tree, altitudes = hg.component_tree_tree_of_shapes_image(image, 'mean', True)
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

    def test_tree_of_shapes_2d_no_immersion_no_padding_original_space(self):
        image = np.asarray(((1, 1, 1, 1, 1),
                            (1, 0, 1, 2, 1),
                            (1, 1, 1, 1, 1)), dtype=np.float64)

        tree, altitudes = hg.component_tree_tree_of_shapes_image(image, 'none', original_size=True, immersion=False)
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

    def test_tree_of_shapes_2d_no_immersion_padding_zero_original_space(self):
        image = np.asarray(((1, 1, 1, 1, 1),
                            (1, 0, 1, 2, 1),
                            (1, 1, 1, 1, 1)), dtype=np.float64)

        tree, altitudes = hg.component_tree_tree_of_shapes_image(image, 'zero', original_size=True, immersion=False)
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

    def test_tree_of_shapes_2d_no_immersion_no_padding_no_original_space(self):
        image = np.asarray(((1, 1, 1, 1, 1),
                            (1, 0, 1, 2, 1),
                            (1, 1, 1, 1, 1)), dtype=np.float64)

        tree, altitudes = hg.component_tree_tree_of_shapes_image(image, 'none', original_size=False, immersion=False)
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

    def test_tree_of_shapes_2d_no_immersion_padding_zero_no_original_space(self):
        image = np.asarray(((1, 1, 1, 1, 1),
                            (1, 0, 1, 2, 1),
                            (1, 1, 1, 1, 1)), dtype=np.float64)

        tree, altitudes = hg.component_tree_tree_of_shapes_image(image, 'zero', original_size=False, immersion=False)
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

    def test_tree_of_shapes_2d_self_dual(self):
        np.random.seed(42)
        image = np.random.rand(25, 38)
        neg_image = -1 * image

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image(image)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image(neg_image)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))

    def test_component_tree_multivariate_tree_of_shapes_image2d_sanity(self):
        image = np.asarray(((1, 1),
                            (1, -2),
                            (1, 7)), dtype=np.float64)

        tree, _ = hg.component_tree_tree_of_shapes_image(image, 'mean')

        image3d = np.dstack((image, image, image))
        tree2 = hg.component_tree_multivariate_tree_of_shapes_image2d(image3d, 'mean')

        self.assertTrue(hg.test_tree_isomorphism(tree, tree2))

    def test_component_tree_multivariate_tree_of_shapes_image2d_1(self):
        im1 = np.asarray((2, 1, 0, 0, -1, -2), dtype=np.float32).reshape((1, 6))
        im2 = np.asarray((1, 2, 0, -2, -1, 0), dtype=np.float32).reshape((1, 6))
        image = np.dstack((im1, im2))

        tree = hg.component_tree_multivariate_tree_of_shapes_image2d(image, 'zero')

        ref_tree = hg.Tree((6, 7, 12, 8, 11, 9, 10, 10, 11, 11, 12, 12, 12))
        self.assertTrue(hg.test_tree_isomorphism(tree, ref_tree))

    def test_component_tree_multivariate_tree_of_shapes_image2d_no_original_size(self):
        im1 = np.asarray((2, 1, 2), dtype=np.float32).reshape((1, 3))
        im2 = np.asarray((2, 2, 1), dtype=np.float32).reshape((1, 3))
        image = np.dstack((im1, im2))

        tree = hg.component_tree_multivariate_tree_of_shapes_image2d(image, 'zero', original_size=False)

        ref_tree = hg.Tree((48, 48, 48, 48, 48, 48, 48, 48, 48,
                            48, 48, 48, 48, 48, 48, 48, 48, 48,
                            48, 48, 45, 45, 45, 47, 46, 48, 48,
                            48, 48, 48, 48, 48, 48, 48, 48, 48,
                            48, 48, 48, 48, 48, 48, 48, 48, 48, 47, 47, 48, 48))
        self.assertTrue(hg.test_tree_isomorphism(tree, ref_tree))

    def test_component_tree_multivariate_tree_of_shapes_image2d_no_padding_no_original_size(self):
        im1 = np.asarray((0, 0, 0, 0, 0,
                          0, 2, 1, 2, 0,
                          0, 0, 0, 0, 0), dtype=np.float32).reshape((3, 5))
        im2 = np.asarray((0, 0, 0, 0, 0,
                          0, 2, 2, 1, 0,
                          0, 0, 0, 0, 0), dtype=np.float32).reshape((3, 5))
        image = np.dstack((im1, im2))

        tree = hg.component_tree_multivariate_tree_of_shapes_image2d(image, 'none', original_size=False)

        ref_tree = hg.Tree((48, 48, 48, 48, 48, 48, 48, 48, 48,
                            48, 48, 48, 48, 48, 48, 48, 48, 48,
                            48, 48, 45, 45, 45, 47, 46, 48, 48,
                            48, 48, 48, 48, 48, 48, 48, 48, 48,
                            48, 48, 48, 48, 48, 48, 48, 48, 48, 47, 47, 48, 48))
        self.assertTrue(hg.test_tree_isomorphism(tree, ref_tree))

    def test_component_tree_multivariate_tree_of_shapes_image2d_no_padding_no_immersion_no_original_size(self):
        im1 = np.asarray((0, 0, 0, 0, 0,
                          0, 2, 1, 0, 0,
                          0, 0, 0, 0, 0), dtype=np.float32).reshape((3, 5))
        im2 = np.asarray((0, 0, 0, 0, 0,
                          0, 0, 1, 2, 0,
                          0, 0, 0, 0, 0), dtype=np.float32).reshape((3, 5))
        image = np.dstack((im1, im2))

        tree = hg.component_tree_multivariate_tree_of_shapes_image2d(image, 'none', original_size=False,
                                                                     immersion=False)

        ref_tree = hg.Tree((18, 18, 18, 18, 18, 18, 15, 17, 16, 18, 18, 18, 18, 18, 18, 17, 17, 18, 18))
        self.assertTrue(hg.test_tree_isomorphism(tree, ref_tree))

    def test_component_tree_multivariate_tree_of_shapes_image2d_no_padding_no_immersion_original_size(self):
        im1 = np.asarray((0, 0, 0, 0, 0,
                          0, 2, 1, 0, 0,
                          0, 0, 0, 0, 0), dtype=np.float32).reshape((3, 5))
        im2 = np.asarray((0, 0, 0, 0, 0,
                          0, 0, 1, 2, 0,
                          0, 0, 0, 0, 0), dtype=np.float32).reshape((3, 5))
        image = np.dstack((im1, im2))

        tree = hg.component_tree_multivariate_tree_of_shapes_image2d(image, 'none', original_size=True, immersion=False)

        ref_tree = hg.Tree((18, 18, 18, 18, 18, 18, 15, 17, 16, 18, 18, 18, 18, 18, 18, 17, 17, 18, 18))
        self.assertTrue(hg.test_tree_isomorphism(tree, ref_tree))

    def test_component_tree_multivariate_tree_of_shapes_image2d_2(self):
        im1 = np.asarray((0, 0, 0, 0, 0, 0, 0,
                          3, 3, 3, 2, 1, 1, 1,
                          3, 3, 3, 2, 1, 1, 1,
                          3, 3, 3, 2, 1, 1, 1,
                          2, 2, 2, 2, 1, 1, 1,
                          1, 1, 1, 1, 1, 0, 0), dtype=np.float32).reshape((6, 7))

        im2 = np.asarray((0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0,
                          0, 2, 1, 1, 1, 2, 0,
                          0, 1, 1, 1, 1, 2, 0,
                          0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0), dtype=np.float32).reshape((6, 7))
        image = np.dstack((im1, im2))

        tree = hg.component_tree_multivariate_tree_of_shapes_image2d(image, 'zero')

        ref_tree = hg.Tree((47, 47, 47, 47, 47, 47, 47,
                            43, 43, 43, 45, 46, 46, 46,
                            43, 42, 43, 45, 45, 44, 46,
                            43, 43, 43, 45, 45, 44, 46,
                            45, 45, 45, 45, 46, 46, 46,
                            46, 46, 46, 46, 46, 47, 47, 43, 45, 45, 46, 47, 47))
        self.assertTrue(hg.test_tree_isomorphism(tree, ref_tree))

    def test_component_tree_multivariate_tree_of_shapes_image2d_fill_hole(self):
        im1 = np.asarray((0, 0, 0, 0, 0, 0, 0,
                          0, 1, 1, 1, 0, 0, 0,
                          0, 1, 0, 0, 0, 0, 0,
                          0, 1, 0, -1, 0, 0, 0,
                          0, 1, 0, 0, 0, 0, 0,
                          0, 1, 1, 1, 0, 0, 0), dtype=np.float32).reshape((6, 7))

        im2 = np.asarray((0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 1, 1, 1, 0,
                          0, 0, 0, 0, 0, 1, 0,
                          0, 0, 0, 0, 0, 1, 0,
                          0, 0, 0, 0, 0, 1, 0,
                          0, 0, 0, 1, 1, 1, 0), dtype=np.float32).reshape((6, 7))
        image = np.dstack((im1, im2))

        tree = hg.component_tree_multivariate_tree_of_shapes_image2d(image, padding='none', immersion=False)

        ref_tree = hg.Tree((44, 44, 44, 44, 44, 44, 44,
                            44, 43, 43, 43, 43, 43, 44,
                            44, 43, 43, 43, 43, 43, 44,
                            44, 43, 43, 42, 43, 43, 44,
                            44, 43, 43, 43, 43, 43, 44,
                            44, 43, 43, 43, 43, 43, 44, 43, 44, 44))

        self.assertTrue(hg.test_tree_isomorphism(tree, ref_tree))

    @staticmethod
    def _expected_leaf_shape(shape, padding, original_size, immersion):
        shape = np.asarray(shape)
        if original_size:
            return tuple(shape)
        if immersion:
            return tuple(shape * 2 - 1) if padding == 'none' else tuple(shape * 2 + 3)
        return tuple(shape) if padding == 'none' else tuple(shape + 2)

    def test_component_tree_multivariate_tree_of_shapes_image2d_leaf_graph(self):
        image = np.arange(24, dtype=np.uint8).reshape((3, 4, 2))
        for padding in ('none', 'zero', 'mean'):
            for immersion in (True, False):
                for original_size in (True, False):
                    with self.subTest(padding=padding, immersion=immersion, original_size=original_size):
                        tree = hg.component_tree_multivariate_tree_of_shapes_image2d(
                            image, padding, original_size=original_size, immersion=immersion)
                        graph = hg.CptHierarchy.get_leaf_graph(tree)
                        self.assertEqual(tree.num_leaves(), graph.num_vertices())
                        self.assertEqual(tuple(hg.CptGridGraph.get_shape(graph)),
                                         self._expected_leaf_shape((3, 4), padding, original_size, immersion))

    def test_component_tree_multivariate_tree_of_shapes_image2d_padding_no_immersion_original_size(self):
        image = np.arange(24, dtype=np.uint8).reshape((3, 4, 2))
        tree = hg.component_tree_multivariate_tree_of_shapes_image2d(image, 'mean', original_size=True,
                                                                     immersion=False)
        self.assertEqual(tree.num_leaves(), 12)

    def test_component_tree_multivariate_tree_of_shapes_image2d_identical_channels(self):
        # with identical channels, the multivariate tree of shapes is the tree of shapes of one channel
        images = (np.asarray(((0, 3, 1, 4),
                              (2, 0, 4, 1),
                              (1, 4, 0, 2)), dtype=np.uint8),
                  np.asarray(((3, 0),
                              (1, 4),
                              (0, 2),
                              (4, 1),
                              (2, 3)), dtype=np.uint8),
                  np.asarray(((2, 1, 0, 0, 1, 2),), dtype=np.float32))
        for image in images:
            for padding in ('none', 'zero', 'mean'):
                for immersion in (True, False):
                    for original_size in (True, False):
                        with self.subTest(shape=image.shape, padding=padding, immersion=immersion,
                                          original_size=original_size):
                            ref_tree, _ = hg.component_tree_tree_of_shapes_image2d(
                                image, padding, original_size=original_size, immersion=immersion)
                            tree = hg.component_tree_multivariate_tree_of_shapes_image2d(
                                np.dstack((image, image)), padding, original_size=original_size,
                                immersion=immersion)
                            self.assertTrue(hg.test_tree_isomorphism(tree, ref_tree))

    # 3d ToS
    def test_tree_of_shapes_3d_self_dual(self):
        np.random.seed(42)
        image = np.random.rand(8, 15, 12)

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image3d(image, 'mean', original_size=True, immersion=True)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image3d(-image, 'mean', original_size=True, immersion=True)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image3d(image, 'mean', original_size=True, immersion=False)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image3d(-image, 'mean', original_size=True, immersion=False)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image3d(image, 'mean', original_size=False, immersion=False)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image3d(-image, 'mean', original_size=False, immersion=False)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image3d(image, 'mean', original_size=False, immersion=False)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image3d(-image, 'mean', original_size=False, immersion=False)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image3d(image, 'none', original_size=True, immersion=True)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image3d(-image, 'none', original_size=True, immersion=True)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image3d(image, 'none', original_size=True, immersion=False)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image3d(-image, 'none', original_size=True, immersion=False)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image3d(image, 'none', original_size=False, immersion=False)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image3d(-image, 'none', original_size=False, immersion=False)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image3d(image, 'none', original_size=False, immersion=False)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image3d(-image, 'none', original_size=False, immersion=False)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))

    def test_tree_of_shapes_3D_flat_no_immersion_no_padding_original_space(self):
        image = np.asarray((((1, 1, 1, 1, 1),
                            (1, 0, 1, 2, 1),
                            (1, 1, 1, 1, 1))), dtype=np.float64).reshape((1, 3, 5))

        tree, altitudes = hg.component_tree_tree_of_shapes_image3d(image, 'none', original_size=False, immersion=False)
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

    def test_tree_of_shapes_3D_flat_default_param(self):
        image = np.asarray((((1, 1),
                            (1, -2),
                            (1, 7))), dtype=np.float64).reshape((1, 3, 2))

        tree, altitudes = hg.component_tree_tree_of_shapes_image3d(image, 'mean', True)
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

    def test_tree_of_shapes_3D_default_param(self):
        image = np.asarray((((1, 1, 1),
                             (1, 1, 1),
                             (1, 1, 1)),
                             ((1, 1, 1),
                             (1, -2, 1),
                             (1, 1, 1)),
                             ((1, 1, 1),
                             (1, 1, 1),
                             (1, 1, 1))), dtype=np.float64).reshape((3, 3, 3))

        tree, altitudes = hg.component_tree_tree_of_shapes_image3d(image, "mean", True)
        ref_parents = np.asarray((28., 28., 28.,
                        28., 28., 28.,
                        28., 28., 28.,
                        28., 28., 28.,
                        28., 27., 28.,
                        28., 28., 28.,
                        28., 28., 28.,
                        28., 28., 28.,
                        28., 28., 28., 28., 28))

        ref_altitudes = np.asarray((1., 1. , 1.,
                          1., 1. , 1.,
                          1., 1. , 1.,
                          1., 1. , 1.,
                          1., -2., 1.,
                          1., 1. , 1.,
                          1., 1. , 1.,
                          1., 1. , 1.,
                          1., 1. , 1., -2., 1.))

        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.allclose(altitudes, ref_altitudes))

    # generic ToS
    def test_tree_of_shapes_self_dual(self):
        np.random.seed(42)
        image = np.random.rand(25, 38, 25)

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image(image, 'mean', original_size=True, immersion=True)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image(-image, 'mean', original_size=True, immersion=True)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image(image, 'mean', original_size=True, immersion=False)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image(-image, 'mean', original_size=True, immersion=False)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image(image, 'mean', original_size=False, immersion=False)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image(-image, 'mean', original_size=False, immersion=False)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image(image, 'mean', original_size=False, immersion=False)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image(-image, 'mean', original_size=False, immersion=False)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image(image, 'none', original_size=True, immersion=True)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image(-image, 'none', original_size=True, immersion=True)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image(image, 'none', original_size=True, immersion=False)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image(-image, 'none', original_size=True, immersion=False)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image(image, 'none', original_size=False, immersion=False)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image(-image, 'none', original_size=False, immersion=False)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))

        tree1, altitudes1 = hg.component_tree_tree_of_shapes_image(image, 'none', original_size=False, immersion=False)
        tree2, altitudes2 = hg.component_tree_tree_of_shapes_image(-image, 'none', original_size=False, immersion=False)

        self.assertTrue(hg.test_tree_isomorphism(tree1, tree2))


    def test_tree_of_shapes_no_immersion_no_padding_original_space(self):
        image = np.asarray((((1, 1, 1, 1, 1),
                            (1, 0, 1, 2, 1),
                            (1, 1, 1, 1, 1))), dtype=np.float64)

        tree, altitudes = hg.component_tree_tree_of_shapes_image(image, 'none', original_size=False, immersion=False)
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

    def test_tree_of_shapes_default_param_2d(self):
        image = np.asarray((((1, 1),
                            (1, -2),
                            (1, 7))), dtype=np.float64)

        tree, altitudes = hg.component_tree_tree_of_shapes_image(image, 'mean', True)
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

    def test_tree_of_shapes_default_param_3d(self):
        image = np.asarray((((1, 1, 1),
                             (1, 1, 1),
                             (1, 1, 1)),
                             ((1, 1, 1),
                             (1, -2, 1),
                             (1, 1, 1)),
                             ((1, 1, 1),
                             (1, 1, 1),
                             (1, 1, 1))))

        tree, altitudes = hg.component_tree_tree_of_shapes_image(image, "mean", True)
        ref_parents = np.asarray((28., 28., 28.,
                        28., 28., 28.,
                        28., 28., 28.,
                        28., 28., 28.,
                        28., 27., 28.,
                        28., 28., 28.,
                        28., 28., 28.,
                        28., 28., 28.,
                        28., 28., 28., 28., 28))

        ref_altitudes = np.asarray((1., 1. , 1.,
                          1., 1. , 1.,
                          1., 1. , 1.,
                          1., 1. , 1.,
                          1., -2., 1.,
                          1., 1. , 1.,
                          1., 1. , 1.,
                          1., 1. , 1.,
                          1., 1. , 1., -2., 1.))

        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.allclose(altitudes, ref_altitudes))

    def test_tree_of_shapes_image2d_3d_forward_arguments(self):
        generic = tos_module.component_tree_tree_of_shapes_image
        signature = inspect.signature(generic)
        sentinel = object()

        for dim, wrapper in ((2, hg.component_tree_tree_of_shapes_image2d),
                             (3, hg.component_tree_tree_of_shapes_image3d)):
            image = np.zeros((2,) * dim, dtype=np.uint8)
            calls = (((image,), {},
                      dict(padding='mean', original_size=True, immersion=True, exterior_vertex=0)),
                     ((image, 'zero', False, False, 7), {},
                      dict(padding='zero', original_size=False, immersion=False, exterior_vertex=7)),
                     ((image,), dict(padding='none', original_size=False, immersion=False, exterior_vertex=3),
                      dict(padding='none', original_size=False, immersion=False, exterior_vertex=3)))
            for args, kwargs, expected in calls:
                with self.subTest(dim=dim, args=args[1:], kwargs=kwargs):
                    with mock.patch.object(tos_module, 'component_tree_tree_of_shapes_image',
                                           return_value=sentinel) as patched:
                        self.assertIs(wrapper(*args, **kwargs), sentinel)
                    patched.assert_called_once()
                    bound = signature.bind(*patched.call_args.args, **patched.call_args.kwargs)
                    bound.apply_defaults()
                    self.assertIs(bound.arguments.pop('image'), image)
                    self.assertEqual(bound.arguments, expected)

    def test_tree_of_shapes_image2d_3d_exterior_vertex(self):
        for dim, wrapper in ((2, hg.component_tree_tree_of_shapes_image2d),
                             (3, hg.component_tree_tree_of_shapes_image3d)):
            image = np.zeros((3,) * dim, dtype=np.uint8)
            image[(1,) * dim] = 10
            # without padding and with immersion, the exterior is given in an image of shape (5,) * dim
            exterior_vertex = int(np.ravel_multi_index((2,) * dim, (5,) * dim))
            options = dict(padding='none', original_size=True, immersion=True)

            tree0, altitudes0 = wrapper(image, exterior_vertex=0, **options)
            tree, altitudes = wrapper(image, exterior_vertex=exterior_vertex, **options)
            ref_tree, ref_altitudes = hg.component_tree_tree_of_shapes_image(image, exterior_vertex=exterior_vertex,
                                                                             **options)
            self.assertEqual(altitudes0[tree0.root()], 0)
            self.assertEqual(altitudes[tree.root()], 10)
            np.testing.assert_array_equal(tree.parents(), ref_tree.parents())
            np.testing.assert_array_equal(altitudes, ref_altitudes)

    def test_tree_of_shapes_image2d_3d_same_as_generic(self):
        for wrapper, image in ((hg.component_tree_tree_of_shapes_image2d,
                                np.asarray(((0, 3, 1, 4),
                                            (2, 0, 4, 1),
                                            (1, 4, 0, 2)), dtype=np.uint8)),
                               (hg.component_tree_tree_of_shapes_image3d,
                                np.arange(24, dtype=np.float32).reshape((2, 3, 4)) % 5)):
            shape = np.asarray(image.shape)
            for padding in ('none', 'zero', 'mean'):
                for immersion in (True, False):
                    for original_size in (True, False):
                        if immersion:
                            work_shape = shape * 2 - 1 if padding == 'none' else shape * 2 + 3
                        else:
                            work_shape = shape if padding == 'none' else shape + 2
                        # default exterior, a vertex near the center and the last vertex of the working space
                        for exterior_vertex in (None, int(np.prod(work_shape)) // 2, int(np.prod(work_shape)) - 1):
                            with self.subTest(wrapper=wrapper.__name__, padding=padding, immersion=immersion,
                                              original_size=original_size, exterior_vertex=exterior_vertex):
                                kwargs = dict(padding=padding, original_size=original_size, immersion=immersion)
                                if exterior_vertex is not None:
                                    kwargs['exterior_vertex'] = exterior_vertex
                                tree, altitudes = wrapper(image, **kwargs)
                                ref_tree, ref_altitudes = hg.component_tree_tree_of_shapes_image(image, **kwargs)
                                np.testing.assert_array_equal(tree.parents(), ref_tree.parents())
                                np.testing.assert_array_equal(altitudes, ref_altitudes)
                                self.assertEqual(tree.category(), ref_tree.category())
                                graph = hg.CptHierarchy.get_leaf_graph(tree)
                                ref_graph = hg.CptHierarchy.get_leaf_graph(ref_tree)
                                self.assertEqual(tuple(hg.CptGridGraph.get_shape(graph)),
                                                 tuple(hg.CptGridGraph.get_shape(ref_graph)))
                                self.assertEqual(tree.num_leaves(), graph.num_vertices())
