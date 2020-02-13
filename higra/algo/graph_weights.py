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


def weight_graph(graph, vertex_weights, weight_function):
    """
    Compute the edge weights of a graph using source and target vertices values
    and specified weighting function (see :class:`~higra.WeightFunction` enumeration).

    :param graph: input graph
    :param vertex_weights: vertex weights of the input graph
    :param weight_function: see :class:`~higra.WeightFunction`
    :return: edge weights of the graph
    """

    vertex_weights = hg.linearize_vertex_weights(vertex_weights, graph)

    edge_weights = hg.cpp._weight_graph(graph, vertex_weights, weight_function)

    return edge_weights
