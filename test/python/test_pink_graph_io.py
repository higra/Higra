import unittest
import numpy as np
import sys

sys.path.insert(0, "@PYTHON_MODULE_PATH@")

import higra as hg

import os
import errno
import os.path


def silentremove(filename):
    try:
        os.remove(filename)
    except:
        pass


class TestPinkGraphIO(unittest.TestCase):

    def test_graphRead(self):
        graph, vertexWeights, edgeWeights, shape = hg.readGraphPink("../ressources/test.graph")

        edgesRef = []
        for i in range(14):
            edgesRef.append((i, i + 1))

        vertexWeightsRef = np.arange(1, 16)
        edgesWeightsRef = (3, 0, 0, 1, 3, 0, 1, 0, 2, 0, 1, 0, 3, 0)

        edges = []
        for e in graph.edges():
            edges.append(e)

        self.assertTrue(shape == [3, 5])
        self.assertTrue(edges == edgesRef)
        self.assertTrue(np.allclose(vertexWeights, vertexWeightsRef))
        self.assertTrue(np.allclose(edgeWeights, edgesWeightsRef))

    def test_graphWrite(self):
        filename = "testWriteGraphPink.graph"
        silentremove(filename)

        vertexWeights = np.arange(1, 16)
        edgesWeights = (3, 0, 0, 1, 3, 0, 1, 0, 2, 0, 1, 0, 3, 0)
        shape = (3, 5)

        graph = hg.UndirectedGraph(15)
        for i in range(14):
            graph.addEdge(i, i + 1)

        hg.saveGraphPink(filename, graph, vertexWeights, edgesWeights, shape)

        self.assertTrue(os.path.exists(filename))

        with open(filename, 'r') as f:
            data = f.read()

        silentremove(filename)

        with open("../ressources/test.graph", 'r') as f:
            dataRef = f.read()

        self.assertTrue(data == dataRef)


if __name__ == '__main__':
    unittest.main()
