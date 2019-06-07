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


def make_region_adjacency_graph_from_labelisation(graph, vertex_labels):
    """
    Create a region adjacency graph (rag) of a vertex labelled graph.
    Each maximal connected set of vertices having the same label is a region.
    Each region is represented by a vertex in the rag.
    There is an edge between two regions of labels :math:`l_1` and :math:`l_2` in the rag iff there exists an edge
    linking two vertices of labels :math:`l_1` and :math:`l_2` int he original graph.

    :param graph: input graph
    :param vertex_labels: vertex labels on the input graph
    :return: a region adjacency graph (Concept :class:`~higra.CptRegionAdjacencyGraph`)
    """
    vertex_labels = hg.linearize_vertex_weights(vertex_labels, graph)

    rag, vertex_map, edge_map = hg.cpp._make_region_adjacency_graph_from_labelisation(graph, vertex_labels)

    hg.CptRegionAdjacencyGraph.link(rag, graph, vertex_map, edge_map)

    return rag


def make_region_adjacency_graph_from_graph_cut(graph, edge_weights):
    """
    Create a region adjacency graph (rag) from a graph cut.
    Two vertices :math:`v_1`, :math:`v_2` are in the same region if there exists a :math:`v_1v_2`-path composed of edges weighted 0.
    Each region is represented by a vertex in the rag.
    There is an edge between two regions of labels :math:`l_1` and :math:`l_2` in the rag iff there exists an edge
    linking two vertices of labels :math:`l_1` and :math:`l_2` int he original graph.

    :param graph: input graph
    :param edge_weights: edge weights on the input graph, non zero weights are part of the cut
    :return: a region adjacency graph (Concept :class:`~higra.CptRegionAdjacencyGraph`)
    """
    rag, vertex_map, edge_map = hg.cpp._make_region_adjacency_graph_from_graph_cut(graph, edge_weights)

    hg.CptRegionAdjacencyGraph.link(rag, graph, vertex_map, edge_map)

    return rag


def rag_back_project_vertex_weights(graph, vertex_weights):
    """
    Projects rag vertex weights onto original graph vertices.
    The result is an array weighting the vertices of the original graph of the rag such that:
    for any vertex :math:`v` of the original graph, its weight is equal to the weight of the vertex of the rag that represents
    the region that contains :math:`v`.

    For any vertex index :math:`i`,
    :math:`result[i] = rag\_vertex\_weight[rag\_vertex_map[i]]`

    :param graph: input region adjacency graph
    :param vertex_weights: vertex weights on the input region adjacency graph
    :return: vertex weights on the original graph
    """

    rag = hg.CptRegionAdjacencyGraph.construct(graph)
    if graph.num_vertices() != vertex_weights.shape[0]:
        raise Exception("vertex_weights size does not match graph size.")

    new_weights = hg.cpp._rag_back_project_weights(rag["vertex_map"], vertex_weights)

    new_weights = hg.delinearize_vertex_weights(new_weights, rag["pre_graph"])

    return new_weights


def rag_back_project_edge_weights(graph, edge_weights):
    """
    Projects rag edge weights onto original graph edges.
    The result is an array weighting the edges of the original graph of the rag such that:
    for any edge :math:`e` of the original graph, its weight is equal to the weight of the edge of the rag that
    represents that links the two regions containing the extremities of :math:`e`. If no such edge exists
    (if the extremities of :math:`e` are in the same region), its value is 0.

    For any edge index :math:`ei`,
    :math:`result[ei] = rag\_edge\_weight[rag\_edge\_map[ei]]` if :math:`rag\_edge\_map[ei] != -1` and 0 otherwise.

    :param graph: input region adjacency graph
    :param edge_weights: edge weights on the input region adjacency graph
    :return: edge weights on the original graph
    """

    rag = hg.CptRegionAdjacencyGraph.construct(graph)
    if graph.num_edges() != edge_weights.shape[0]:
        raise Exception("edge_weights size does not match graph size.")

    new_weights = hg.cpp._rag_back_project_weights(rag["edge_map"], edge_weights)

    return new_weights


@hg.argument_helper(hg.CptRegionAdjacencyGraph)
def rag_accumulate_on_vertices(rag, accumulator, vertex_weights):
    """
    Weights rag vertices by accumulating values from the vertex weights of the original graph.

    For any vertex index :math:`i` of the rag,
    :math:`result[i] = accumulator(\{vertex\_weights[j] | rag\_vertex\_map[j] == i\})`

    :param rag: input region adjacency graph (Concept :class:`~higra.RegionAdjacencyGraph`)
    :param vertex_weights: vertex weights on the original graph
    :param accumulator: see :class:`~higra.Accumulators`
    :return: vertex weights on the region adjacency graph
    """

    detail = hg.CptRegionAdjacencyGraph.construct(rag)
    vertex_weights = hg.linearize_vertex_weights(vertex_weights, detail["pre_graph"])

    new_weights = hg.cpp._rag_accumulate(detail["vertex_map"], vertex_weights, accumulator)

    return new_weights


@hg.argument_helper(hg.CptRegionAdjacencyGraph)
def rag_accumulate_on_edges(rag, accumulator, edge_weights):
    """
    Weights rag edges by accumulating values from the edge weights of the original graph.

    For any edge index :math:`ei` of the rag,
    :math:`result[ei] = accumulate(\{edge\_weights[j] | rag\_edge\_map[j] == ei\})`

    :param rag: input region adjacency graph (Concept :class:`~higra.RegionAdjacencyGraph`)
    :param edge_weights: edge weights on the original graph
    :param accumulator: see :class:`~higra.Accumulators`
    :return: edge weights on the region adjacency graph
    """

    detail = hg.CptRegionAdjacencyGraph.construct(rag)

    new_weights = hg.cpp._rag_accumulate(detail["edge_map"], edge_weights, accumulator)

    return new_weights
