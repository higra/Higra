############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Wonder Alexandre Luz Alves                              #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import unittest
import higra as hg
import numpy as np


def naive_area_casf(graph, image, thresholds):
    current = image
    for threshold in thresholds:
        max_tree, max_altitudes = hg.component_tree_max_tree(graph, current)
        max_area = hg.attribute_area(max_tree)
        filtered = hg.reconstruct_leaf_data(max_tree, max_altitudes, max_area <= threshold)

        min_tree, min_altitudes = hg.component_tree_min_tree(graph, filtered)
        min_area = hg.attribute_area(min_tree)
        current = hg.reconstruct_leaf_data(min_tree, min_altitudes, min_area <= threshold)

    return current


class TestComponentTreeDualFilter(unittest.TestCase):

    def test_connected_alternating_sequential_filter_area(self):
        graph = hg.get_4_adjacency_implicit_graph((6, 6))
        image = np.asarray(((-5, 2, 2, 5, 5, 5),
                            (-4, 2, 2, 6, 5, 4),
                            (3, 3, 3, 3, 3, 3),
                            (-2, -2, -2, 9, 7, 6),
                            (-1, 0, -2, 8, 9, 8),
                            (-1, -1, -2, 7, 8, 9)), dtype=np.float64)
        thresholds = [2, 4, 6]

        expected = naive_area_casf(graph, image, thresholds)
        result = hg.connected_alternating_sequential_filter(graph, image, "area", thresholds)

        self.assertTrue(np.allclose(result, expected))

    def test_connected_alternating_sequential_filter_accepts_enum(self):
        graph = hg.get_4_adjacency_implicit_graph((4, 4))
        image = np.asarray(((0, 2, 0, 1),
                            (2, 5, 2, 1),
                            (0, 2, 4, 3),
                            (1, 1, 3, 6)), dtype=np.uint8)
        thresholds = [1, 2, 3]

        result_str = hg.connected_alternating_sequential_filter(graph, image, "area", thresholds)
        result_enum = hg.connected_alternating_sequential_filter(graph, image, hg.CasfAttribute.area, thresholds)

        self.assertTrue(np.allclose(result_str, result_enum))

    def test_connected_alternating_sequential_filter_empty_thresholds(self):
        graph = hg.get_4_adjacency_implicit_graph((3, 4))
        image = np.asarray(((0, 1, 2, 3),
                            (4, 5, 6, 7),
                            (8, 9, 10, 11)), dtype=np.uint8)

        result = hg.connected_alternating_sequential_filter(graph, image, "area", [])

        self.assertTrue(np.allclose(result, image))

    def test_connected_alternating_sequential_filter_preserves_shape(self):
        graph = hg.get_4_adjacency_implicit_graph((3, 5))
        image = np.asarray(((0, 1, 2, 3, 4),
                            (5, 6, 7, 8, 9),
                            (10, 11, 12, 13, 14)), dtype=np.uint8)

        result = hg.connected_alternating_sequential_filter(graph, image, "area", [1, 2, 3])

        self.assertEqual(result.shape, image.shape)

    def test_connected_alternating_sequential_filter_invalid_attribute(self):
        graph = hg.get_4_adjacency_implicit_graph((3, 3))
        image = np.asarray(((0, 1, 2),
                            (3, 4, 5),
                            (6, 7, 8)), dtype=np.uint8)

        with self.assertRaises(ValueError):
            hg.connected_alternating_sequential_filter(graph, image, "invalid_attribute", [1, 2, 3])

    def test_connected_alternating_sequential_filter_bounding_box(self):
        graph = hg.get_4_adjacency_implicit_graph((5, 6))
        image = np.asarray(((0, 0, 0, 0, 0, 0),
                            (0, 5, 5, 0, 3, 0),
                            (0, 5, 5, 0, 3, 0),
                            (0, 0, 0, 0, 3, 0),
                            (0, 0, 0, 0, 0, 0)), dtype=np.uint8)

        result = hg.connected_alternating_sequential_filter(graph, image, "bounding_box_width", [1, 2, 3])

        self.assertEqual(result.shape, image.shape)

    def test_connected_alternating_sequential_filter_bounding_box_requires_embedding(self):
        graph = hg.UndirectedGraph(4)
        graph.add_edges((0, 1, 0, 2), (1, 3, 2, 3))
        image = np.asarray((0, 1, 2, 3), dtype=np.uint8)

        with self.assertRaises(RuntimeError):
            hg.connected_alternating_sequential_filter(graph, image, "bounding_box_width", [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
