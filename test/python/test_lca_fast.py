import unittest
import numpy as np
import sys

sys.path.insert(0, "@PYTHON_MODULE_PATH@")

import higra as hg


class TestLCAFast(unittest.TestCase):

    @staticmethod
    def getTree():
        parentRelation = np.asarray((5, 5, 6, 6, 6, 7, 7, 7), dtype=np.uint64)
        return hg.Tree(parentRelation)

    def test_LCAFast(self):
        t = TestLCAFast.getTree()
        lca = hg.LCAFast(t)

        self.assertTrue(lca.lca(0, 0) == 0)
        self.assertTrue(lca.lca(3, 3) == 3)
        self.assertTrue(lca.lca(5, 5) == 5)
        self.assertTrue(lca.lca(7, 7) == 7)
        self.assertTrue(lca.lca(0, 1) == 5)
        self.assertTrue(lca.lca(1, 0) == 5)
        self.assertTrue(lca.lca(2, 3) == 6)
        self.assertTrue(lca.lca(2, 4) == 6)
        self.assertTrue(lca.lca(3, 4) == 6)
        self.assertTrue(lca.lca(5, 6) == 7)
        self.assertTrue(lca.lca(0, 2) == 7)
        self.assertTrue(lca.lca(1, 4) == 7)
        self.assertTrue(lca.lca(2, 6) == 6)

    def test_LCAFastV(self):
        g = hg._get4AdjacencyGraph((2, 2))
        t = hg.Tree((4, 4, 5, 5, 6, 6, 6))
        lca = hg.LCAFast(t)

        ref = (4, 6, 6, 5)
        res = lca.lca(g)
        self.assertTrue(np.allclose(ref, res))


if __name__ == '__main__':
    unittest.main()
