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


@hg.argument_helper(hg.CptEdgeWeightedGraph, ("graph", hg.CptGridGraph))
def graph_4_adjacency_2_khalimsky(edge_weights, graph, shape, add_extra_border=False):
    """
    Create a contour image in the Khalimsky grid from a 4 adjacency edge-weighted graph.

    :param edge_weights: edge weights of the graph (Concept :class:`~higra.CptEdgeWeightedGraph`)
    :param graph: must be a 4 adjacency 2d graph (deduced from :class:`~higra.CptEdgeWeightedGraph`, Concept :class:`~higra.CptGridGraph`)
    :param shape: shape of the graph (deduced from :class:`~higra.CptGridGraph`)
    :param add_extra_border: if False result size is 2 * shape - 1 and 2 * shape + 1 otherwise
    :return: a 2d array
    """
    shape = hg.normalize_shape(shape)
    return hg._graph_4_adjacency_2_khalimsky(graph, shape, edge_weights, add_extra_border)


def khalimsky_2_graph_4_adjacency(khalimsky, extra_border=False):
    """
    Create a 4 adjacency edge-weighted graph from a contour image in the Khalimsky grid.

    :param khalimsky: a 2d array
    :param extra_border: if False the shape of the Khalimsky image  is 2 * shape - 1 and 2 * shape + 1 otherwise, where shape is the shape of the resulting grid graph
    :return: a graph (Concept :class:`~higra.CptGridGraph`) and its edge weights (Concept :class:`~higra.CptEdgeWeightedGraph`)
    """

    graph, embedding, edge_weights = hg._khalimsky_2_graph_4_adjacency(khalimsky, extra_border)

    hg.CptEdgeWeightedGraph.link(edge_weights, graph)
    hg.CptGridGraph.link(graph, embedding.shape())

    return graph, edge_weights


def get_4_adjacency_graph(shape):
    """
    Create an explicit undirected 4 adjacency graph of the given shape.

    :param shape: a pair (height, width)
    :return: a graph (Concept :class:`~higra.CptGridGraph`)
    """
    graph = hg._get_4_adjacency_graph(shape)
    hg.CptGridGraph.link(graph, shape)
    return graph


def get_8_adjacency_graph(shape):
    """
    Create an explicit undirected 8 adjacency graph of the given shape.

    :param shape: a pair (height, width)
    :return: a graph (Concept :class:`~higra.CptGridGraph`)
    """
    graph = hg._get_8_adjacency_graph(shape)
    hg.CptGridGraph.link(graph, shape)
    return graph


def get_4_adjacency_implicit_graph(shape):
    """
    Create an implicit undirected 4 adjacency graph of the given shape (edges are not stored).

    :param shape: a pair (height, width)
    :return: a graph (Concept :class:`~higra.CptGridGraph`)
    """
    graph = hg._get_4_adjacency_implicit_graph(shape)
    hg.CptGridGraph.link(graph, shape)
    return graph


def get_8_adjacency_implicit_graph(shape):
    """
    Create an implicit undirected 8 adjacency graph of the given shape (edges are not stored).

    :param shape: a pair (height, width)
    :return: a graph (Concept :class:`~higra.CptGridGraph`)
    """
    graph = hg._get_8_adjacency_implicit_graph(shape)
    hg.CptGridGraph.link(graph, shape)
    return graph