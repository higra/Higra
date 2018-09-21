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


@hg.data_consumer(vertex_labels="vertex_labels")
def labelisation_2_graph_cut(graph, vertex_labels):
    """
    Determine the graph cut that corresponds to a given labeling
    of the graph vertices.

    The result is a weighting of the graph edges where edges with
    a non zero weight are part of the cut.

    :param graph:
    :param vertex_labels:
    :return:
    """
    return hg._labelisation_2_graph_cut(graph, vertex_labels)


@hg.data_consumer(edge_weights="edge_weights")
def graph_cut_2_labelisation(graph, edge_weights):
    """
    Labelize graph vertices according to the given graph cut.

    Each edge having a non zero value in the given edge_weights
    are assumed to be part of the cut.

    :param graph:
    :param edge_weights:
    :return:
    """
    return hg._graph_cut_2_labelisation(graph, edge_weights)
