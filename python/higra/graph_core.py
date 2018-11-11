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

    :param graph:
    :param vertex_labels:
    :return:
    """
    vertex_labels = hg.linearize_vertex_weights(vertex_labels, graph)
    graph_cut = hg._labelisation_2_graph_cut(graph, vertex_labels)
    hg.CptEdgeWeightedGraph.link(graph_cut, graph)
    return graph_cut


@hg.argument_helper(hg.CptGraphCut)
def graph_cut_2_labelisation(edge_weights, graph):
    """
    Labelize graph vertices according to the given graph cut.

    Each edge having a non zero value in the given edge_weights
    are assumed to be part of the cut.

    :param graph:
    :param edge_weights:
    :return:
    """
    vertex_labels = hg._graph_cut_2_labelisation(graph, edge_weights)

    vertex_labels = hg.delinearize_vertex_weights(vertex_labels, graph)
    hg.CptVertexLabeledGraph.link(vertex_labels, graph)

    return vertex_labels
