import unittest
import higra as hg
import numpy as np
import math


class TestGraphWeights(unittest.TestCase):

    def test_weighting_graph(self):
        g = hg._get_4_adjacency_graph((2, 2))
        data = np.asarray((0, 1, 2, 3))

        ref = (0.5, 1, 2, 2.5)
        r = hg._weight_graph(g, data, hg.WeightFunction.mean)
        self.assertTrue(np.allclose(ref, r))

        ref = (0, 0, 1, 2)
        r = hg._weight_graph(g, data, hg.WeightFunction.min)
        self.assertTrue(np.allclose(ref, r))

        ref = (1, 2, 3, 3)
        r = hg._weight_graph(g, data, hg.WeightFunction.max)
        self.assertTrue(np.allclose(ref, r))

        ref = (1, 2, 2, 1)
        r = hg._weight_graph(g, data, hg.WeightFunction.L1)
        self.assertTrue(np.allclose(ref, r))

        ref = (math.sqrt(1), 2, 2, math.sqrt(1))
        r = hg._weight_graph(g, data, hg.WeightFunction.L2)
        self.assertTrue(np.allclose(ref, r))

        ref = (1, 2, 2, 1)
        r = hg._weight_graph(g, data, hg.WeightFunction.L_infinity)
        self.assertTrue(np.allclose(ref, r))

        ref = (1, 4, 4, 1)
        r = hg._weight_graph(g, data, hg.WeightFunction.L2_squared)
        self.assertTrue(np.allclose(ref, r))

    def test_weighting_graph_vectorial(self):
        g = hg._get_4_adjacency_graph((2, 2))
        data = np.asarray(((0, 1), (2, 3), (4, 5), (6, 7)))

        ref = (4, 8, 8, 4)
        r = hg._weight_graph(g, data, hg.WeightFunction.L1)
        self.assertTrue(np.allclose(ref, r))

        ref = (math.sqrt(8), math.sqrt(32), math.sqrt(32), math.sqrt(8))
        r = hg._weight_graph(g, data, hg.WeightFunction.L2)
        self.assertTrue(np.allclose(ref, r))

        ref = (2, 4, 4, 2)
        r = hg._weight_graph(g, data, hg.WeightFunction.L_infinity)
        self.assertTrue(np.allclose(ref, r))

        ref = (8, 32, 32, 8)
        r = hg._weight_graph(g, data, hg.WeightFunction.L2_squared)
        self.assertTrue(np.allclose(ref, r))

    def test_weighting_graph_lambda(self):
        g = hg._get_4_adjacency_graph((2, 2))
        data = np.asarray((0, 1, 2, 3))

        ref = (1, 2, 4, 5)
        r = hg._weight_graph(g, lambda i, j: i + j)
        self.assertTrue(np.allclose(ref, r))

        ref = (1, 2, 4, 5)
        r = hg._weight_graph(g, lambda i, j: data[i] + data[j])
        self.assertTrue(np.allclose(ref, r))


if __name__ == '__main__':
    unittest.main()
