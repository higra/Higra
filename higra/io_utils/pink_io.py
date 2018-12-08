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
import numpy as np


def read_graph_pink(filename):
    """
    Read a graph file stored in pink ascii format
    :param filename:
    :return: Graph (with attributes "vertex_weights", "edge_weights", "shape")
    """
    graph, vertex_weights, edge_weights, shape = hg._read_graph_pink(filename)

    hg.CptGridGraph.link(graph, shape)
    vertex_weights = hg.delinearize_vertex_weights(vertex_weights, graph, shape)
    hg.CptVertexWeightedGraph.link(vertex_weights, graph)
    hg.CptEdgeWeightedGraph.link(edge_weights, graph)

    return graph, vertex_weights, edge_weights


@hg.argument_helper(("edge_weights", hg.CptEdgeWeightedGraph), ("vertex_weights", hg.CptVertexWeightedGraph), ("graph", hg.CptGridGraph))
def save_graph_pink(filename, graph, vertex_weights=None, edge_weights=None, shape=None):
    """
    Save a (vertex/edge weighted) graph in the pink ascii file format
    :param filename:
    :param graph:
    :param edge_weights:
    :param vertex_weights:
    :param shape:
    :return:
    """

    if edge_weights is None:
        edge_weights = np.ones((graph.num_edges(),), dtype=np.np.float64)

    if vertex_weights is None:
        vertex_weights = np.ones((graph.num_vertices(),), dtype=np.float64)

    if shape is None:
        shape = (graph.num_vertices(), 1)

    vertex_weights = hg.linearize_vertex_weights(vertex_weights, graph, shape)

    hg._save_graph_pink(filename, graph, vertex_weights=vertex_weights, edge_weights=edge_weights, shape=shape)
