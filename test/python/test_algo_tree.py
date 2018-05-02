import unittest

import sys

sys.path.insert(0, "@PYTHON_MODULE_PATH@")

import higra as hg
import numpy as np


class TestAlgorithmTree(unittest.TestCase):

    def test_tree_isomorphism(self):
        t1 = hg.Tree(np.asarray((5, 5, 6, 6, 7, 8, 7, 8, 8)))
        t2 = hg.Tree(np.asarray((6, 6, 5, 5, 7, 7, 8, 8, 8)))
        t3 = hg.Tree(np.asarray((7, 7, 5, 5, 6, 6, 8, 8, 8)))

        self.assertTrue(hg.test_tree_isomorphism(t1, t2))
        self.assertTrue(hg.test_tree_isomorphism(t2, t1))
        self.assertTrue(hg.test_tree_isomorphism(t1, t3))
        self.assertTrue(hg.test_tree_isomorphism(t3, t1))
        self.assertTrue(hg.test_tree_isomorphism(t2, t3))
        self.assertTrue(hg.test_tree_isomorphism(t3, t2))

        t4 = hg.Tree(np.asarray((5, 5, 7, 6, 6, 8, 7, 8, 8)))

        self.assertTrue(not hg.test_tree_isomorphism(t1, t4))
        self.assertTrue(not hg.test_tree_isomorphism(t2, t4))
        self.assertTrue(not hg.test_tree_isomorphism(t3, t4))
        self.assertTrue(not hg.test_tree_isomorphism(t4, t1))
        self.assertTrue(not hg.test_tree_isomorphism(t4, t2))
        self.assertTrue(not hg.test_tree_isomorphism(t4, t3))


if __name__ == '__main__':
    unittest.main()
