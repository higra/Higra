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


class TestContour2d(unittest.TestCase):

    @staticmethod
    def contour_2_khalimsky(graph, shape, contour):
        res_shape = (shape[0] * 2 - 1, shape[1] * 2 - 1)
        result = np.zeros(res_shape, np.int64)

        embedding = hg.EmbeddingGrid2d(shape)
        count = 0

        def edge_to_k(edge_index):
            e = graph.edge_from_index(edge_index)
            s = e[0]
            t = e[1]

            ti = embedding.lin2grid(t)
            si = embedding.lin2grid(s)
            return ti + si

        for polyline in contour:
            for segment in polyline:
                count += 1
                for e in segment:
                    p = edge_to_k(e[0])
                    result[p[0], p[1]] = count

                p = edge_to_k(segment[0][0])
                result[p[0], p[1]] = -1 * count
                p = edge_to_k(segment[len(segment) - 1][0])
                result[p[0], p[1]] = -1 * count

        return result

    def test_fit_contour_2d(self):
        shape = (4, 5)
        g = hg.get_4_adjacency_graph(shape)

        data = np.asarray((0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0), np.int32)

        ref = np.asarray(((0, 0, 0, 0, 0, 0, 0, 0, 0),
                          (1, 0, 0, 0, 5, 0, 0, 0, 7),
                          (0, 2, 0, 4, 0, 6, 0, 8, 0),
                          (0, 0, 3, 0, 0, 0, 7, 0, 0),
                          (0, 0, 0, 0, 0, 0, 0, 0, 0),
                          (0, 0, 0, 0, 0, 0, 0, 0, 0),
                          (0, 0, 0, 0, 0, 0, 0, 0, 0)), np.int32)

        contours = hg.fit_contour_2d(g, shape, data)
        contours.subdivide(0.000001, False, 0)
        contours_khalimsky = TestContour2d.contour_2_khalimsky(g, shape, contours)
        self.assertTrue(hg.is_in_bijection(ref, contours_khalimsky))

    def test_rag_2d_vertex_perimeter_and_edge_length_simple(self):

        shape = (3, 2)
        g = hg.get_4_adjacency_graph(shape)

        vertex_labels = np.array((1, 2,
                                  3, 3,
                                  4, 4))

        rag = hg.make_region_adjacency_graph_from_labelisation(g, vertex_labels)
        vertex_perimeter, edge_length = hg.rag_2d_vertex_perimeter_and_edge_length(rag)

        ref_perimeter = (4, 4, 6, 6)
        ref_length = (1, 1, 1, 2)

        self.assertTrue(np.all(vertex_perimeter == ref_perimeter))
        self.assertTrue(np.all(edge_length == ref_length))


if __name__ == '__main__':
    unittest.main()
