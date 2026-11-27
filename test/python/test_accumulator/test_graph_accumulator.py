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


class TestGraphAccumulator(unittest.TestCase):

    def test_accumulate_graph_vertices(self):
        g = hg.get_4_adjacency_graph((2, 3))
        vertex_weights = np.asarray((1, 2, 3, 4, 5, 6))

        res1 = hg.accumulate_graph_vertices(g, vertex_weights, hg.Accumulators.max)
        ref1 = (4, 5, 6, 5, 6, 5)
        self.assertTrue(np.all(res1 == ref1))

        vertex_weights2 = np.asarray(((1, 6),
                                      (2, 5),
                                      (3, 4),
                                      (4, 3),
                                      (5, 2),
                                      (6, 1)))

        res2 = hg.accumulate_graph_vertices(g , vertex_weights2, hg.Accumulators.sum)
        ref2 = np.asarray(((6, 8),
                           (9, 12),
                           (8, 6),
                           (6, 8),
                           (12, 9),
                           (8, 6)))
        self.assertTrue(np.all(res2 == ref2))

    def test_accumulate_graph_edges(self):
        g = hg.get_4_adjacency_graph((2, 3))

        edge_weights = np.asarray((1, 2, 3, 4, 6, 5, 7))

        res1 = hg.accumulate_graph_edges(g , edge_weights, hg.Accumulators.max)
        ref1 = (2, 4, 6, 5, 7, 7)
        self.assertTrue(np.all(res1 == ref1))

        edge_weights2 = np.asarray(((1, 6),
                                    (2, 5),
                                    (3, 4),
                                    (4, 3),
                                    (5, 2),
                                    (6, 1),
                                    (7, 9)))
        res2 = hg.accumulate_graph_edges(g, edge_weights2, hg.Accumulators.sum)
        ref2 = np.asarray(((3, 11),
                           (8, 13),
                           (8, 6),
                           (8, 6),
                           (17, 13),
                           (12, 11)))
        self.assertTrue(np.all(res2 == ref2))

    def test_accumulate_graph_mean(self):
        g = hg.get_4_adjacency_graph((2, 3))
        edge_weights = np.asarray((1, 2, 3, 4, 6, 5, 7), dtype=np.float64)
        res = hg.accumulate_graph_edges(g, edge_weights, hg.Accumulators.mean)
        ref = hg.accumulate_graph_edges(g, edge_weights, hg.Accumulators.sum) / \
              hg.accumulate_graph_edges(g, edge_weights, hg.Accumulators.counter)
        np.testing.assert_allclose(res, ref)
        self.assertEqual(res[0], 1.5)

        vertex_weights = np.asarray(((1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)), dtype=np.float64)
        res = hg.accumulate_graph_vertices(g, vertex_weights, hg.Accumulators.mean)
        np.testing.assert_allclose(res, ((3, 4), (3, 4), (4, 3), (3, 4), (4, 3), (4, 3)))

    def test_accumulate_graph_empty_neighbourhood_and_result_types(self):
        g = hg.UndirectedGraph(3)
        g.add_edge(0, 1)
        vertex_weights = np.asarray((1., 2., 3.))
        np.testing.assert_array_equal(hg.accumulate_graph_vertices(g, vertex_weights, hg.Accumulators.first), (2, 1, 0))
        np.testing.assert_array_equal(hg.accumulate_graph_vertices(g, vertex_weights, hg.Accumulators.last), (2, 1, 0))
        np.testing.assert_array_equal(hg.accumulate_graph_edges(g, np.asarray((4.,)), hg.Accumulators.first), (4, 4, 0))

        vertex_weights2 = np.asarray(((1, 2), (3, 4), (5, 6)), dtype=np.uint8)
        res = hg.accumulate_graph_vertices(g, vertex_weights2, hg.Accumulators.argmin)
        self.assertEqual(res.dtype, np.int64)
        np.testing.assert_array_equal(res, (0, 0, -1))
        res = hg.accumulate_graph_vertices(g, vertex_weights2, hg.Accumulators.counter)
        self.assertEqual(res.dtype, np.int64)
        np.testing.assert_array_equal(res, (1, 1, 0))


if __name__ == '__main__':
    unittest.main()
