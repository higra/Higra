import unittest
import numpy as np
import sys

sys.path.insert(0, "@PYTHON_MODULE_PATH@")

import higra as hg
import numpy as np


class TestHierarchyCore(unittest.TestCase):

    @staticmethod
    def getTree():
        parentRelation = np.asarray((5, 5, 6, 6, 6, 7, 7, 7), dtype=np.uint64)
        return hg.Tree(parentRelation)

    def test_BPTTrivial(self):
        graph = hg._get4AdjacencyGraph((1, 2))

        edgeWeights = np.asarray((2,))

        tree, altitudes, mst = hg._bptCanonical(graph, edgeWeights)

        self.assertTrue(tree.numVertices() == 3)
        self.assertTrue(tree.numEdges() == 2)
        self.assertTrue(np.allclose(tree.parents(), (2, 2, 2)))
        self.assertTrue(np.allclose(altitudes, (0, 0, 2)))
        self.assertTrue(mst.numVertices() == 2)
        self.assertTrue(mst.numEdges() == 1)

    def test_BPT(self):
        graph = hg._get4AdjacencyGraph((2, 3))

        edgeWeights = np.asarray((1, 0, 2, 1, 1, 1, 2))

        tree, altitudes, mst = hg._bptCanonical(graph, edgeWeights)
        self.assertTrue(tree.numVertices() == 11)
        self.assertTrue(tree.numEdges() == 10)
        self.assertTrue(np.allclose(tree.parents(), (6, 7, 9, 6, 8, 9, 7, 8, 10, 10, 10)))
        self.assertTrue(np.allclose(altitudes, (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2)))
        self.assertTrue(mst.numVertices() == 6)
        self.assertTrue(mst.numEdges() == 5)

        ref = [(0, 3),
               (0, 1),
               (1, 4),
               (2, 5),
               (1, 2)]
        test = []
        for e in mst.edges():
            test.append(e)
        self.assertTrue(ref == test)

    def test_simplifyTree(self):
        t = TestHierarchyCore.getTree()

        altitudes = np.asarray((0, 0, 0, 0, 0, 1, 2, 2))

        criterion = np.equal(altitudes, altitudes[t.parents()])

        newTree, nodeMap = hg._simplifyTree(t, criterion)

        # for reference
        newAltitudes = altitudes[nodeMap]

        self.assertTrue(newTree.numVertices() == 7)

        refp = np.asarray((5, 5, 6, 6, 6, 6, 6))
        self.assertTrue(np.all(refp == newTree.parents()))

        refnm = np.asarray((0, 1, 2, 3, 4, 5, 7))
        self.assertTrue(np.all(refnm == nodeMap))


if __name__ == '__main__':
    unittest.main()
