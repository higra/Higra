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


class TestGraphImage(unittest.TestCase):

    def test_graph_4_adjacency_2_khalimsky(self):
        g = hg.get_4_adjacency_graph((2, 3))
        data = np.asarray((1, 0, 2, 1, 1, 1, 2))

        ref = np.asarray(((0, 1, 0, 2, 0),
                          (0, 1, 1, 2, 1),
                          (0, 1, 0, 2, 0)))
        r = hg.graph_4_adjacency_2_khalimsky(g, data)
        self.assertTrue(np.allclose(ref, r))

    def test_khalimsky_2_graph_4_adjacency(self):
        data = np.asarray((1, 0, 2, 1, 1, 1, 2))

        ref = np.asarray(((0, 1, 0, 2, 0),
                          (0, 1, 1, 2, 1),
                          (0, 1, 0, 2, 0)))
        graph, weights = hg.khalimsky_2_graph_4_adjacency(ref)
        shape = hg.CptGridGraph.get_shape(graph)
        self.assertTrue(np.allclose(shape, (2, 3)))
        self.assertTrue(np.allclose(data, weights))

        _, weights2 = hg.khalimsky_2_graph_4_adjacency(ref, graph=graph)
        self.assertTrue(np.allclose(data, weights2))

    def test_get_4_adjacency_graph(self):
        shape = (2, 3)
        graph = hg.get_4_adjacency_graph(shape)

        self.assertTrue(graph.num_vertices() == 6)

        res_edges = set(zip(*graph.edge_list()))
        ref_edges = set(((0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)))

        self.assertTrue(ref_edges == res_edges)

    def test_get_8_adjacency_graph(self):
        shape = (2, 3)
        graph = hg.get_8_adjacency_graph(shape)

        self.assertTrue(graph.num_vertices() == 6)

        res_edges = set(zip(*graph.edge_list()))
        ref_edges = set(((0, 1), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (4, 5)))

        self.assertTrue(ref_edges == res_edges)

    def test_mask_2_neighbours(self):
        mask = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

        res = hg.mask_2_neighbours(mask)
        ref = ((-1, 0), (0, -1), (0, 1), (1, 0))
        self.assertTrue(set(ref) == set(tuple(map(tuple, res))))

        center = [1, 1]
        res = hg.mask_2_neighbours(mask, center)
        ref = ((-1, 0), (0, -1), (0, 1), (1, 0))
        self.assertTrue(set(ref) == set(tuple(map(tuple, res))))

        center = [2, 1]
        res = hg.mask_2_neighbours(mask, center)
        ref = ((-2, 0), (-1, -1), (-1, 1), (0, 0))
        self.assertTrue(set(ref) == set(tuple(map(tuple, res))))

        mask = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
        res = hg.mask_2_neighbours(mask)
        ref = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
        self.assertTrue(set(ref) == set(tuple(map(tuple, res))))

    def test_get_nd_regular_graph(self):
        mask = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
        neighbours = hg.mask_2_neighbours(mask)
        shape = (2, 3, 2)
        graph = hg.get_nd_regular_graph(shape, neighbours)

        self.assertTrue(graph.num_vertices() == 12)

        ref_edges = set(((0, 1), (0, 2), (0, 6), (1, 3), (1, 7), (2, 3), (2, 4), (2, 8), (3, 5), (3, 9), (4, 5),
                         (4, 10), (5, 11), (6, 7), (6, 8), (7, 9), (8, 9), (8, 10), (9, 11), (10, 11)))
        res_edges = set(zip(*graph.edge_list()))

        self.assertTrue(ref_edges == res_edges)

    def test_match_pixels_image_2d(self):
        im1 = np.asarray([[1, 0, 0, 1],
                          [0, 0, 0, 1]])
        im2 = np.asarray([[0, 0, 0, 1],
                          [0, 0, 1, 0]])

        sources, targets = hg.match_pixels_image_2d(im1, im2, 1.3, "absolute")

        ref_edges = set(((3, 3), (7, 6)))
        for e in zip(sources, targets):
            e = tuple(e)
            self.assertTrue(e in ref_edges)
            ref_edges.remove(e)


    def test_match_pixels_image_2d_2(self):
        im1 = np.asarray([[1, 0, 0, 1, 1], #3, 2;  4, 3
                          [0, 0, 0, 1, 0], #8, 7
                          [0, 0, 0, 1, 0], #13, 12
                          [0, 0, 1, 1, 1]]) #17, 17; 18,18; 19, 14
        im2 = np.asarray([[0, 0, 1, 1, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 1],
                          [1, 1, 1, 1, 0]])

        sources, targets = hg.match_pixels_image_2d(im1, im2, 1.3, "absolute")
        ref_edges = set(((3, 2), (4, 3), (8, 7), (13, 12), (17, 17), (18, 18), (19, 14)))
        for e in zip(sources, targets):
            e = tuple(e)
            self.assertTrue(e in ref_edges)
            ref_edges.remove(e)





if __name__ == '__main__':
    unittest.main()
