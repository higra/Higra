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


@hg.argument_helper(hg.CptVertexLabeledGraph)
def labelisation_2_graph_cut(vertex_labels, graph):
    """
    Determine the graph cut that corresponds to a given labeling
    of the graph vertices.

    The result is a weighting of the graph edges where edges with
    a non zero weight are part of the cut.

    :param vertex_labels: Weights on the vertices of the graph (Concept :class:`~higra.CptVertexLabeledGraph`)
    :param graph: input graph (deduced from :class:`~higra.CptVertexLabeledGraph`)
    :return: graph edge-weights representing the equivalent cut (Concept :class:`~higra.CptGraphCut`)
    """
    vertex_labels = hg.linearize_vertex_weights(vertex_labels, graph)
    graph_cut = hg.cpp._labelisation_2_graph_cut(graph, vertex_labels)
    hg.CptGraphCut.link(graph_cut, graph)
    return graph_cut


@hg.argument_helper(hg.CptGraphCut)
def graph_cut_2_labelisation(edge_weights, graph):
    """
    Labelise graph vertices according to the given graph cut.

    Each edge having a non zero value in the given edge_weights
    are assumed to be part of the cut.

    :param edge_weights: Weights on the edges of the graph (Concept :class:`~higra.CptEdgeWeightedGraph`)
    :param graph: Input graph (deduced from :class:`~higra.CptEdgeWeightedGraph`)
    :return: A labelisation of the graph vertices (Concept :class:`~higra.CptVertexWeightedGraph`)
    """
    vertex_labels = hg.cpp._graph_cut_2_labelisation(graph, edge_weights)

    vertex_labels = hg.delinearize_vertex_weights(vertex_labels, graph)
    hg.CptVertexLabeledGraph.link(vertex_labels, graph)

    return vertex_labels


@hg.argument_helper(hg.CptEdgeWeightedGraph)
def undirected_graph_2_adjacency_matrix(edge_weights, graph, non_edge_value=0):
    """
    Create an adjacency matrix from an undirected edge-weighted graph (the result is thus symmetric).

    As the given graph is not necessarily complete, non-existing edges will receive the value `non_edge_value` in
    the adjacency matrix.

    :param edge_weights: Graph edge weights (Concept :class:`~higra.CptEdgeWeightedGraph`)
    :param graph: Input graph (deduced from :class:`~higra.CptEdgeWeightedGraph`)
    :param non_edge_value: Value used to represent edges that are not in the input graph
    :return: A 2d symmetric square matrix
    """
    return hg.cpp._undirected_graph_2_adjacency_matrix(graph, edge_weights, float(non_edge_value))


def adjacency_matrix_2_undirected_graph(adjacency_matrix, non_edge_value=0):
    """
    Creates an undirected edge-weighted graph from an adjacency matrix.

    Adjacency matrix entries which are equal to `non_edge_value` are not considered to be part of the graph.

    :param adjacency_matrix: Input adjacency matrix (A 2d symmetric square matrix)
    :param non_edge_value: Value used to represent non existing edges in the adjacency matrix
    :return: a pair (UndirectedGraph, ndarray) representing the graph and its edge_weights (Concept :class:`~higra.CptEdgeWeightedGraph`)
    """
    graph, edge_weights = hg.cpp._adjacency_matrix_2_undirected_graph(adjacency_matrix, float(non_edge_value))
    hg.CptEdgeWeightedGraph.link(edge_weights, graph)
    return graph, edge_weights


@hg.argument_helper(hg.CptEdgeWeightedGraph)
def ultrametric_open(edge_weights, graph):
    """
    Computes the subdominant ultrametric of the given edge weighted graph.

    The subdominant ultrametric relative to a given dissimilarity measure (here the graph edge weights)
    is defined as the largest ultrametric smaller than the dissimilarity measure.

    In the case of an edge weighted undirected graph, the value of the subdominant ultrametric on the edge `{x,y}`
    is given by the min-max distance between `x` and `y`.

    Complexity: `O(n*log(n))` with `n` the number of edges in the graph

    :param edge_weights: Graph edge weights (Concept :class:`~higra.CptEdgeWeightedGraph`)
    :param graph: Input graph (deduced from :class:`~higra.CptEdgeWeightedGraph`)
    :return: edge weights corresponding to the subdominant ultrametric (Concept :class:`~higra.CptEdgeWeightedGraph`)
    """
    tree, altitudes = hg.bpt_canonical(edge_weights, graph)
    return hg.saliency(altitudes)


@hg.argument_helper(hg.CptEdgeWeightedGraph)
def minimum_spanning_tree(edge_weights, graph):
    """
    Computes the minimum spanning tree of the given edge weighted graph with Kruskal's algorithm.

    If the input graph is not connected, the result is indeed a minimum spanning forest.

    Complexity: `O(n*log(n))` with `n` the number of edges in the graph

    :param edge_weights: Graph edge weights (Concept :class:`~higra.CptEdgeWeightedGraph`)
    :param graph: Input graph (deduced from :class:`~higra.CptEdgeWeightedGraph`)
    :return: a minimum spanning tree of the input edge wieghted graph (Concept :class:`~higra.CptMinimumSpanningTree`)
    """
    mst, edge_map = hg.cpp._minimum_spanning_tree(graph, edge_weights)
    hg.CptMinimumSpanningTree.link(mst, mst_edge_map=edge_map, base_graph=graph)
    return mst