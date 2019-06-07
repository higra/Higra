############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import higra as hg
import numpy as np


def labelisation_2_graph_cut(graph, vertex_labels):
    """
    Determines the graph cut that corresponds to a given labeling
    of the graph vertices.

    The result is a weighting of the graph edges where edges with
    a non zero weight are part of the cut.

    :param graph: input graph
    :param vertex_labels: Weights on the vertices of the graph
    :return: graph edge-weights representing the equivalent cut
    """
    vertex_labels = hg.linearize_vertex_weights(vertex_labels, graph)
    graph_cut = hg.cpp._labelisation_2_graph_cut(graph, vertex_labels)

    return graph_cut


def graph_cut_2_labelisation(graph, edge_weights):
    """
    Labelises graph vertices according to the given graph cut.

    Each edge having a non zero value in the given edge_weights
    are assumed to be part of the cut.

    :param graph: Input graph
    :param edge_weights: Weights on the edges of the graph
    :return: A labelisation of the graph vertices
    """
    vertex_labels = hg.cpp._graph_cut_2_labelisation(graph, edge_weights)

    vertex_labels = hg.delinearize_vertex_weights(vertex_labels, graph)

    return vertex_labels


def undirected_graph_2_adjacency_matrix(graph, edge_weights=None, non_edge_value=0):
    """
    Adjacency matrix corresponding to an undirected edge-weighted graph (the result is thus symmetric).

    As the given graph is not necessarily complete, non-existing edges will receive the value :attr:`non_edge_value` in
    the adjacency matrix.

    :param graph: Input graph
    :param edge_weights: Graph edge weights (default to ``np.ones(graph.num_edges())`` if ``None``)
    :param non_edge_value: Value used to represent edges that are not in the input graph
    :return: A 2d symmetric square matrix
    """
    if edge_weights is None:
        edge_weights = np.ones((graph.num_edges(),), np.float32)
    return hg.cpp._undirected_graph_2_adjacency_matrix(graph, edge_weights, float(non_edge_value))


def adjacency_matrix_2_undirected_graph(adjacency_matrix, non_edge_value=0):
    """
    Undirected edge-weighted graph corresponding to an adjacency matrix.

    Adjacency matrix entries which are equal to :attr:`non_edge_value` are not considered to be part of the graph.

    :param adjacency_matrix: Input adjacency matrix (A 2d symmetric square matrix)
    :param non_edge_value: Value used to represent non existing edges in the adjacency matrix
    :return: a pair (UndirectedGraph, ndarray) representing the graph and its edge_weights (Concept :class:`~higra.CptEdgeWeightedGraph`)
    """
    return hg.cpp._adjacency_matrix_2_undirected_graph(adjacency_matrix, float(non_edge_value))


def ultrametric_open(graph, edge_weights):
    """
    Subdominant ultrametric of the given edge weighted graph.

    The subdominant ultrametric relative to a given dissimilarity measure (here the graph edge weights)
    is defined as the largest ultrametric smaller than the dissimilarity measure.

    In the case of an edge weighted undirected graph, the value of the subdominant ultrametric on the
    edge :math:`e_{xy}` is given by the min-max distance between :math:`x` and :math:`y`.

    Complexity: :math:`\mathcal{O}(n*log(n))` with :math:`n` the number of edges in the graph

    :param graph: Input graph
    :param edge_weights: Graph edge weights
    :return: edge weights corresponding to the subdominant ultrametric
    """
    tree, altitudes = hg.bpt_canonical(edge_weights, graph)
    return hg.saliency(altitudes)


def minimum_spanning_tree(graph, edge_weights):
    """
    Computes the minimum spanning tree of the given edge weighted graph with Kruskal's algorithm.

    If the input graph is not connected, the result is indeed a minimum spanning forest.

    Complexity: :math:`\mathcal{O}(n*log(n))` with :math:`n` the number of edges in the graph

    :param graph: Input graph
    :param edge_weights: Graph edge weights
    :return: a minimum spanning tree of the input edge weighted graph (Concept :class:`~higra.CptMinimumSpanningTree`)
    """
    mst, edge_map = hg.cpp._minimum_spanning_tree(graph, edge_weights)
    hg.CptMinimumSpanningTree.link(mst, mst_edge_map=edge_map, base_graph=graph)
    return mst
