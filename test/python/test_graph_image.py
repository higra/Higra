import unittest

import sys

sys.path.insert(0, "@PYTHON_MODULE_PATH@")

import higra as hg
import numpy as np


class TestGraphImage(unittest.TestCase):

    def test_contour_2_khalimsky(self):
        g = hg._get4AdjacencyGraph((2, 3))
        data = np.asarray((1, 0, 2, 1, 1, 1, 2))

        ref = np.asarray(((0, 1, 0, 2, 0),
                          (0, 1, 1, 2, 1),
                          (0, 1, 0, 2, 0)))
        r = hg._contour2Khalimsky(g, (2, 3), data)
        self.assertTrue(np.allclose(ref, r))

    def test_khalimsky_2_contour(self):
        data = np.asarray((1, 0, 2, 1, 1, 1, 2))

        ref = np.asarray(((0, 1, 0, 2, 0),
                          (0, 1, 1, 2, 1),
                          (0, 1, 0, 2, 0)))
        graph, embedding, weights = hg._khalimsky2Contour(ref)
        self.assertTrue(np.allclose(embedding.shape(), (2, 3)))
        self.assertTrue(np.allclose(data, weights))


if __name__ == '__main__':
    unittest.main()
