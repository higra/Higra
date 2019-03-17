############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import unittest
import higra as hg
import numpy as np


class TestAlgorithmGraphCore(unittest.TestCase):

    def test_graph_cut_2_labelisation(self):
        graph = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0), dtype=np.int32)

        labels = hg.graph_cut_2_labelisation(edge_weights, graph)

        ref_labels = np.asarray((1, 2, 2, 1, 1, 3, 1, 3, 3), dtype=np.int32)
        self.assertTrue(hg.is_in_bijection(labels, ref_labels))

    def test_labelisation_2_graph_cut(self):
        graph = hg.get_4_adjacency_graph((3, 3))
        labels = np.asarray((1, 2, 2, 1, 1, 3, 1, 3, 3), dtype=np.int32)

        edge_weights = hg.labelisation_2_graph_cut(labels, graph)

        ref_edge_weights = np.asarray((1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0), dtype=np.int32)

        self.assertTrue(hg.is_in_bijection(edge_weights, ref_edge_weights))

    def test_undirected_graph_2_adjacency_matrix(self):
        graph = hg.UndirectedGraph(5)
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(0, 3)
        graph.add_edge(0, 4)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(2, 4)

        edge_weights = np.asarray((1, 2, 3, 4, 5, 6, 7))
        adj_mat = hg.undirected_graph_2_adjacency_matrix(edge_weights, graph, -1)

        ref_adj_mat = np.asarray(((-1, 1, 2, 3, 4),
                                  (1, -1, 5, -1, -1),
                                  (2, 5, -1, 6, 7),
                                  (3, -1, 6, -1, -1),
                                  (4, -1, 7, -1, -1)))
        self.assertTrue(np.all(ref_adj_mat == adj_mat))

        t = hg.Tree(np.asarray((5, 5, 6, 6, 6, 7, 7, 7)))
        edge_weights = np.asarray((1, 2, 3, 4, 5, 6, 7))
        adj_mat = hg.undirected_graph_2_adjacency_matrix(edge_weights, t)

        ref_adj_mat = np.asarray(((0, 0, 0, 0, 0, 1, 0, 0),
                                  (0, 0, 0, 0, 0, 2, 0, 0),
                                  (0, 0, 0, 0, 0, 0, 3, 0),
                                  (0, 0, 0, 0, 0, 0, 4, 0),
                                  (0, 0, 0, 0, 0, 0, 5, 0),
                                  (1, 2, 0, 0, 0, 0, 0, 6),
                                  (0, 0, 3, 4, 5, 0, 0, 7),
                                  (0, 0, 0, 0, 0, 6, 7, 0)))
        self.assertTrue(np.all(ref_adj_mat == adj_mat))

    def test_undirected_graph_2_adjacency_matrix_overload_resolution(self):
        graph = hg.UndirectedGraph(2)
        graph.add_edge(0, 1)

        edge_weights = np.asarray((0.1,), dtype=np.float64)
        adj_mat = hg.undirected_graph_2_adjacency_matrix(edge_weights, graph)

        ref_adj_mat = np.asarray(((0, 0.1),
                                  (0.1, 0)))
        self.assertTrue(np.all(ref_adj_mat == adj_mat))
        self.assertTrue(adj_mat.dtype == np.float64)

    def test_adjacency_matrix_2_undirected_graph(self):
        ref_adj_mat = np.asarray(((0, 0.1),
                                  (0.1, 0)), dtype=np.float64)
        graph, edge_weights = hg.adjacency_matrix_2_undirected_graph(ref_adj_mat)

        ref_graph = hg.UndirectedGraph(2)
        ref_graph.add_edge(0, 1)

        ref_edge_weights = np.asarray((0.1,))

        self.assertTrue(edge_weights.dtype == np.float64)
        self.assertTrue(np.all(edge_weights == ref_edge_weights))
        self.assertTrue(graph.num_vertices() == ref_graph.num_vertices())
        self.assertTrue(graph.num_edges() == ref_graph.num_edges())

        for (e1, e2) in zip(graph.edges(), ref_graph.edges()):
            self.assertTrue(e1 == e2)

    def test_adjacency_matrix_2_undirected_graph_overload_resolution(self):
        ref_adj_mat = np.asarray(((-1, 1, 2, 3, 4),
                                  (1, -1, 5, -1, -1),
                                  (2, 5, -1, 6, 7),
                                  (3, -1, 6, -1, -1),
                                  (4, -1, 7, -1, -1)))
        graph, edge_weights = hg.adjacency_matrix_2_undirected_graph(ref_adj_mat, -1)

        ref_graph = hg.UndirectedGraph(5)
        ref_graph.add_edge(0, 1)
        ref_graph.add_edge(0, 2)
        ref_graph.add_edge(0, 3)
        ref_graph.add_edge(0, 4)
        ref_graph.add_edge(1, 2)
        ref_graph.add_edge(2, 3)
        ref_graph.add_edge(2, 4)

        ref_edge_weights = np.asarray((1, 2, 3, 4, 5, 6, 7))

        self.assertTrue(np.all(edge_weights == ref_edge_weights))
        self.assertTrue(graph.num_vertices() == ref_graph.num_vertices())
        self.assertTrue(graph.num_edges() == ref_graph.num_edges())

        for (e1, e2) in zip(graph.edges(), ref_graph.edges()):
            self.assertTrue(e1 == e2)

    def ultrametric_open(self):
        graph = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((2, 3, 9, 5, 10, 1, 5, 8, 2, 2, 4, 3), dtype=np.int32)

        subd_ultrametric = hg.ultrametric_open(edge_weights, graph)

        ref_subd_ultrametric = np.asarray((2, 3, 9, 3, 9, 1, 4, 3, 2, 2, 4, 3), dtype=np.int32)

        self.assertTrue(hg.is_in_bijection(subd_ultrametric, ref_subd_ultrametric))

    def test_minimum_spanning_tree(self):
        graph = hg.get_4_adjacency_graph((2, 3))

        edge_weights = np.asarray((1, 0, 2, 1, 1, 1, 2))

        mst = hg.minimum_spanning_tree(edge_weights, graph)
        mst_edge_map = hg.CptMinimumSpanningTree.get_edge_map(mst)

        self.assertTrue(mst.num_vertices() == 6)
        self.assertTrue(mst.num_edges() == 5)

        ref_sources = (0, 0, 1, 2, 1)
        ref_targets = (3, 1, 4, 5, 2)
        sources, targets = mst.edge_list()

        self.assertTrue(np.all(sources == ref_sources))
        self.assertTrue(np.all(targets == ref_targets))

        self.assertTrue(np.all(mst_edge_map == (1, 0, 3, 4, 2)))

    def test_minimum_spanning_forest(self):
        graph = hg.UndirectedGraph(6)
        graph.add_edges((0, 0, 1, 3, 3, 4), (1, 2, 2, 4, 5, 5))

        edge_weights = np.asarray((0, 1, 2, 3, 4, 5))

        mst = hg.minimum_spanning_tree(edge_weights, graph)
        mst_edge_map = hg.CptMinimumSpanningTree.get_edge_map(mst)

        self.assertTrue(mst.num_vertices() == 6)
        self.assertTrue(mst.num_edges() == 4)

        ref_sources = (0, 0, 3, 3)
        ref_targets = (1, 2, 4, 5)
        sources, targets = mst.edge_list()

        self.assertTrue(np.all(sources == ref_sources))
        self.assertTrue(np.all(targets == ref_targets))

        self.assertTrue(np.all(mst_edge_map == (0, 1, 3, 4)))


if __name__ == '__main__':
    unittest.main()
