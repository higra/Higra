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


def accumulate_graph_edges(graph, edge_weights, accumulator):
    """
    For each vertex i of the graph: accumulates edge weights of the out edges of i and put the result
    in output. i.e. output(i) = accumulate(edge_weights(out_edges(i)))

    :param graph: input graph
    :param edge_weights: Weights on the edges of the graph
    :param accumulator: see :class:`~higra.Accumulators`
    :return: returns new graph vertex weights
    """
    res = hg.cpp._accumulate_graph_edges(graph, edge_weights, accumulator)
    return res


def accumulate_graph_vertices(graph, vertex_weights, accumulator):
    """
    For each vertex i of the graph: accumulates vertex weights of the adjacent vertices of i and put the result
    in output. i.e. output(i) = accumulate(vertex_weights(adjacent_vertices(i)))

    :param graph: input graph
    :param vertex_weights: Weights on the vertices of the graph
    :param accumulator: see :class:`~higra.Accumulators`
    :return: returns new graph vertex weights
    """
    res = hg.cpp._accumulate_graph_vertices(graph, vertex_weights, accumulator)
    return res
