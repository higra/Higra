import unittest

import sys

sys.path.insert(0, "@PYTHON_MODULE_PATH@")

import higra as hg
import numpy as np


class TestGraphImage(unittest.TestCase):

    def test_contour_2_khalimsky(self):
        g = hg.get4AdjacencyGraph((2, 3))
        data = np.asarray((1, 0, 2, 1, 1, 1, 2))

        ref = np.asarray(((0, 1, 0, 2, 0),
                          (0, 1, 1, 2, 1),
                          (0, 1, 0, 2, 0)))
        r = hg.contour2Khalimsky(g, hg.EmbeddingGrid2d((2, 3)), data)
        self.assertTrue(np.allclose(ref, r))


if __name__ == '__main__':
    unittest.main()
