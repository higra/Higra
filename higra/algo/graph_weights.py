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


@hg.argument_helper(hg.CptVertexWeightedGraph)
def weight_graph(vertex_weights, weight_function, graph):
    """
    Compute the edge weights of a graph using source and target vertices values
    and specified weighting function (see :class:`~higra.WeightFunction` enumeration).

    :param vertex_weights: vertex weights of the input graph  (Concept :class:`~higra.CptVertexWeightedGraph`)
    :param weight_function: see :class:`~higra.WeightFunction`
    :param graph: input graph (deduced from :class:`~higra.CptVertexWeightedGraph`)
    :return: edge weights of the graph (Concept :class:`~higra.CptEdgeWeightedGraph`)
    """

    vertex_weights = hg.linearize_vertex_weights(vertex_weights, graph)

    edge_weights = hg.cpp._weight_graph(graph, vertex_weights, weight_function)

    hg.CptEdgeWeightedGraph.link(edge_weights, graph)

    return edge_weights


def weight_graph_function(graph, weight_function):
    """
    Compute the edge weights of a graph with the given weighting function.
    The weighting function takes the vertex index of the extremities of an edge and returns the weight of the edge (scalar)

    Set the attribute "edge_weights" of graph with the computed values.

    :param graph: input graph
    :param weight_function: eg. lambda i, j: abs(data[i]-[j])
    :return: edge weights of the graph (Concept :class:`~higra.CptEdgeWeightedGraph`)
    """

    edge_weights = hg.cpp._weight_graph(graph, weight_function)

    hg.CptEdgeWeightedGraph.link(edge_weights, graph)

    return edge_weights
