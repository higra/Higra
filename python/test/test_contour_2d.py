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
            e = graph.edge(edge_index)
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

    @staticmethod
    def is_in_bijection(a, b):
        
        aa = a.flatten()
        bb = b.flatten()

        if aa.size != bb.size:
            return False

        equiv1 = {}
        equiv2 = {}

        for i in range(aa.size):
            v1 = aa[i]
            v2 = bb[i]
            
            if v1 in equiv1:
                if equiv1[v1] != v2:
                    return False
            else:
                equiv1[v1] = v2

            if v2 in equiv2:
                if equiv2[v2] != v1:
                    return False
            else:
                equiv2[v2] = v1

        return True

    def test_fit_contour_2d(self):
        shape = (4, 5)
        g = hg.get_4_adjacency_graph(shape)

        data = np.asarray((0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0), np.int32)

        ref  = np.asarray( ((0, 0, 0, 0, 0, 0, 0, 0, 0),
                            (1, 0, 0, 0, 5, 0, 0, 0, 7),
                            (0, 2, 0, 4, 0, 6, 0, 8, 0),
                            (0, 0, 3, 0, 0, 0, 7, 0, 0),
                            (0, 0, 0, 0, 0, 0, 0, 0, 0),
                            (0, 0, 0, 0, 0, 0, 0, 0, 0),
                            (0, 0, 0, 0, 0, 0, 0, 0, 0)), np.int32)

        contours = hg.fit_contour_2d(g, shape, data)
        contours.subdivide(0.000001, False, 0)
        contours_khalimsky = TestContour2d.contour_2_khalimsky(g, shape, contours)
        self.assertTrue(TestContour2d.is_in_bijection(ref, contours_khalimsky))

if __name__ == '__main__':
    unittest.main()
