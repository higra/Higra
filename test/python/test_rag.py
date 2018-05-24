import sys
import unittest

import numpy as np

sys.path.insert(0, "@PYTHON_MODULE_PATH@")

import higra as hg


class TestRag(unittest.TestCase):

    @staticmethod
    def get_rag():
        g = hg.get_4_adjacency_graph((4, 4))
        vertex_labels = np.asarray((1, 1, 5, 5,
                                    1, 1, 5, 5,
                                    1, 1, 3, 3,
                                    1, 1, 10, 10))

        return hg._make_region_adjacency_graph(g, vertex_labels)

    def test_make_rag(self):
        rag, vertex_map, edge_map = TestRag.get_rag()

        self.assertTrue(rag.num_vertices() == 4)
        self.assertTrue(rag.num_edges() == 5)

        expected_edges = ((0, 1), (1, 2), (0, 2), (2, 3), (0, 3))

        i = 0
        for e in rag.edges():
            self.assertTrue(e == expected_edges[i])
            i += 1

        expected_vertex_map = np.asarray((0, 0, 1, 1,
                                          0, 0, 1, 1,
                                          0, 0, 2, 2,
                                          0, 0, 3, 3))
        self.assertTrue(np.allclose(vertex_map, expected_vertex_map))

        iv = -1
        expected_edge_map = np.asarray((iv, iv, 0, iv, iv, iv, iv,
                                        iv, iv, 0, iv, iv, 1, 1,
                                        iv, iv, 2, iv, iv, 3, 3,
                                        iv, 4, iv))
        self.assertTrue(np.allclose(edge_map, expected_edge_map))

    def test_back_project_vertex(self):
        rag, vertex_map, edge_map = TestRag.get_rag()

        rag_vertex_weights = np.asarray((5, 7, 1, 3))
        vertex_weights = hg._rag_back_project_weights(vertex_map, rag_vertex_weights)
        expected_vertex_weights = np.asarray((5, 5, 7, 7,
                                              5, 5, 7, 7,
                                              5, 5, 1, 1,
                                              5, 5, 3, 3))
        self.assertTrue(np.allclose(vertex_weights, expected_vertex_weights))

    def test_back_project_edge(self):
        rag, vertex_map, edge_map = TestRag.get_rag()

        rag_edge_weights = np.asarray((5, 7, 1, 3, 2))
        edge_weights = hg._rag_back_project_weights(edge_map, rag_edge_weights)
        iv = 0
        expected_edge_weights = np.asarray((iv, iv, 5, iv, iv, iv, iv,
                                            iv, iv, 5, iv, iv, 7, 7,
                                            iv, iv, 1, iv, iv, 3, 3,
                                            iv, 2, iv))
        self.assertTrue(np.allclose(edge_weights, expected_edge_weights))

    def test_accumulate_vertex(self):
        rag, vertex_map, edge_map = TestRag.get_rag()
        vertex_weights = np.ones((16,))

        rag_vertex_weights = hg._rag_accumulate(vertex_map, vertex_weights, hg.Accumulators.sum)

        expected_rag_vertex_weights = (8, 4, 2, 2)

        self.assertTrue(np.allclose(rag_vertex_weights, expected_rag_vertex_weights))

    def test_accumulate_edge(self):
        rag, vertex_map, edge_map = TestRag.get_rag()
        edge_weights = np.ones((24,))

        rag_edge_weights = hg._rag_accumulate(edge_map, edge_weights, hg.Accumulators.sum)

        expected_rag_edge_weights = (2, 2, 1, 2, 1)

        self.assertTrue(np.allclose(rag_edge_weights, expected_rag_edge_weights))


if __name__ == '__main__':
    unittest.main()
