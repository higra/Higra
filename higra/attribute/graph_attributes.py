############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################


import numpy as np
import higra as hg


@hg.auto_cache
def attribute_vertex_area(graph):
    """
    Vertex area of the given graph.

    In general the area of a vertex if simply equal to 1. But, if the graph is a region adjacency graph then the area of
    a region is equal to the sum of the area of the vertices inside the region (obtained with a recursive call to
    ``attribute_vertex_area`` on the original graph).

    :param graph: input graph
    :return: a 1d array
    """
    if hg.CptRegionAdjacencyGraph.validate(graph):  # this is a rag like graph
        pre_graph = hg.CptRegionAdjacencyGraph.get_pre_graph(graph)
        pre_graph_vertex_area = attribute_vertex_area(pre_graph)
        return hg.rag_accumulate_on_vertices(graph, hg.Accumulators.sum, vertex_weights=pre_graph_vertex_area)
    res = np.ones((graph.num_vertices(),), dtype=np.float64)
    res = hg.delinearize_vertex_weights(res, graph)
    return res


@hg.auto_cache
def attribute_edge_length(graph):
    """
    Edge length of the given graph.

    In general the length of an edge if simply equal to 1. But, if the graph is a region adjacency graph then the
    length of an edge is equal to the sum of length of the corresponding edges in the original graph (obtained with a
    recursive call to ``attribute_edge_length`` on the original graph).

    :param graph: input graph
    :return: a nd array
    """
    if hg.CptRegionAdjacencyGraph.validate(graph):  # this is a rag like graph
        pre_graph = hg.CptRegionAdjacencyGraph.get_pre_graph(graph)
        pre_graph_edge_length = attribute_edge_length(pre_graph)
        return hg.rag_accumulate_on_edges(graph, hg.Accumulators.sum, edge_weights=pre_graph_edge_length)
    res = np.ones((graph.num_edges(),), dtype=np.float64)
    return res


@hg.auto_cache
def attribute_vertex_perimeter(graph, edge_length=None):
    """
    Vertex perimeter of the given graph.
    The perimeter of a vertex is defined as the sum of the length of out-edges of the vertex.

    If the input graph has an attribute value `no_border_vertex_out_degree`, then each vertex perimeter is assumed to be
    equal to this attribute value. This is a convenient method to handle image type graphs where an outer border has to be
    considered.

    :param graph: input graph
    :param edge_length: length of the edges of the input graph (provided by :func:`~higra.attribute_edge_length` on `graph`)
    :return: a nd array
    """
    if edge_length is None:
        edge_length = hg.attribute_edge_length(graph)

    special_case_border_graph = hg.get_attribute(graph, "no_border_vertex_out_degree")

    if special_case_border_graph is not None:
        res = np.full((graph.num_vertices(),), special_case_border_graph, dtype=np.float64)
        res = hg.delinearize_vertex_weights(res, graph)
        return res

    res = hg.accumulate_graph_edges(graph, edge_length, hg.Accumulators.sum)
    res = hg.delinearize_vertex_weights(res, graph)
    return res


@hg.argument_helper(hg.CptGridGraph)
@hg.auto_cache
def attribute_vertex_coordinates(graph, shape):
    """
    Coordinates of the vertices of the given grid graph.

    Example
    =======

    >>> g = hg.get_4_adjacency_graph((2, 3))
    >>> c = hg.attribute_vertex_coordinates(g)
    (((0, 0), (0, 1), (0, 2)),
     ((1, 0), (1, 1), (1, 2)))

    :param graph: Input graph (Concept :class:`~higra.CptGridGraph`)
    :param shape: (deduced from :class:`~higra.CptGridGraph`)
    :return: a nd array
    """
    coords = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coords = [c.reshape((-1,)) for c in coords]
    attribute = np.stack(list(reversed(coords)), axis=1)
    attribute = hg.delinearize_vertex_weights(attribute, graph)
    return attribute
