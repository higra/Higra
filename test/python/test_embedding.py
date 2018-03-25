import unittest
import numpy as np
import sys

sys.path.insert(0, "@PYTHON_MODULE_PATH@")

import higra as hg


class TestEmbedding(unittest.TestCase):

    def test_createEmbedding(self):
        e1 = hg.EmbeddingGrid((3, 5, 2))
        e2 = hg.EmbeddingGrid([3, 5, 2])
        e3 = hg.EmbeddingGrid(np.asarray([3, 5, 2]))


if __name__ == '__main__':
    unittest.main()
