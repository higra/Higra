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

@hg.argument_helper(hg.CptEdgeWeightedGraph)
def accumulate_graph_edges(edge_weights, accumulator, graph):
    """
    For each vertex i of the graph: accumulates edge weights of the out edges of i and put the result
    in output. i.e. output(i) = accumulate(edge_weights(out_edges(i)))

    :param edge_weights: Weights on the edges of the graph (Concept :class:`~higra.CptEdgeWeightedGraph`)
    :param accumulator: see :class:`~higra.Accumulators`
    :param graph: input graph (deduced from :class:`~higra.CptEdgeWeightedGraph`)
    :return: returns new graph vertex weights (Concept :class:`~higra.CptEdgeWeightedGraph`)
    """
    res = hg.cpp._accumulate_graph_edges(graph, edge_weights, accumulator)
    hg.CptVertexWeightedGraph.link(res, graph)
    return res


@hg.argument_helper(hg.CptVertexWeightedGraph)
def accumulate_graph_vertices(vertex_weights, accumulator, graph):
    """
    For each vertex i of the graph: accumulates vertex weights of the adjacent vertices of i and put the result
    in output. i.e. output(i) = accumulate(vertex_weights(adjacent_vertices(i)))

    :param vertex_weights: Weights on the vertices of the graph (Concept :class:`~higra.CptEdgeWeightedGraph`)
    :param accumulator: see :class:`~higra.Accumulators`
    :param graph: input graph (deduced from :class:`~higra.CptEdgeWeightedGraph`)
    :return: returns new graph vertex weights (Concept :func:`~higra.CptEdgeWeightedGraph`)
    """
    res = hg.cpp._accumulate_graph_vertices(graph, vertex_weights, accumulator)
    hg.CptVertexWeightedGraph.link(res, graph)
    return res

