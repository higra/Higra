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
    and specified weighting function (see WeightFunction enumeration).

    Set the attribute "vertex_weights" of graph with the provided values.
    Set the attribute "edge_weights" of graph with the computed values.

    :param graph:
    :param vertex_weights:
    :param weight_function: in WeightFunction enumeration
    :return: edge_weights
    """

    vertex_weights = hg.linearize_vertex_weights(vertex_weights, graph)

    edge_weights = hg._weight_graph(graph, vertex_weights, weight_function)

    hg.CptEdgeWeightedGraph.link(edge_weights, graph)

    return edge_weights


def weight_graph_function(graph, weight_function):
    """
    Compute the edge weights of a graph with the given weighting function.
    The weighting function takes the vertex index of the extremities of an edge and returns the weight of the edge (scalar)

    Set the attribute "edge_weights" of graph with the computed values.

    :param graph:
    :param weight_function: eg. lambda i, j: abs(data[i]-[j])
    :return: edge_weights
    """

    edge_weights = hg._weight_graph(graph, weight_function)

    hg.CptEdgeWeightedGraph.link(edge_weights, graph)

    return edge_weights
