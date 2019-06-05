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
    Accumulates edge weights of the out edges of each vertex :math:`i`:
    ie :math:`output(i) = accumulator(edge\_weights(out\_edges(i)))`.

    :param graph: input graph
    :param edge_weights: Weights on the edges of the graph
    :param accumulator: see :class:`~higra.Accumulators`
    :return: returns new graph vertex weights
    """
    res = hg.cpp._accumulate_graph_edges(graph, edge_weights, accumulator)
    return res


def accumulate_graph_vertices(graph, vertex_weights, accumulator):
    """
    Accumulates vertex weights of the adjacent vertices of each vertex :math:`i`:
    ie :math:`output(i) = accumulator(vertex\_weights(adjacent\_vertices(i)))`.

    :param graph: input graph
    :param vertex_weights: Weights on the vertices of the graph
    :param accumulator: see :class:`~higra.Accumulators`
    :return: returns new graph vertex weights
    """
    res = hg.cpp._accumulate_graph_vertices(graph, vertex_weights, accumulator)
    return res
