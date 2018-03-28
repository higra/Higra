import unittest
import numpy as np
import sys

sys.path.insert(0, "@PYTHON_MODULE_PATH@")

import higra as hg


def compare(t1, t2):
    if len(t1) != len(t2):
        return False
    for i in range(len(t1)):
        if t1[i] != t2[i]:
            return False
    return True


class TestEmbedding(unittest.TestCase):

    def test_createEmbedding(self):
        e1 = hg.EmbeddingGrid((3, 5, 2))
        e2 = hg.EmbeddingGrid([3, 5, 2])
        e3 = hg.EmbeddingGrid(np.asarray([3, 5, 2], dtype=np.int64))

        for e in (e1, e2, e3):
            self.assertTrue(e.dimension() == 3)
            self.assertTrue(e.size() == 30)
            self.assertTrue(compare(e.shape(), (3, 5, 2)))


if __name__ == '__main__':
    unittest.main()
