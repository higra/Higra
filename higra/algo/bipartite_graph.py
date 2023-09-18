############################################################################
# Copyright ESIEE Paris (2023)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import higra as hg
import numpy as np


def is_bipartite_graph(graph, return_coloring=True):
    """
    Test if a graph is bipartite and return a coloring if it is bipartite.

    A bipartite graph is a graph whose vertices can be divided into two disjoint and independent sets
    :math:`X` and :math:`Y` such that every edge connects a vertex in :math:`X` to one in :math:`Y`.

    If the graph is bipartite, the function returns the value ``True`` and an array of colors.
    For any vertex :math:`v`, :math:`color[v] == 0` if :math:`v` belongs to :math:`X` and
    :math:`color[v] == 1` if :math:`v` belongs to :math:`Y`.
    Note that the coloring is not unique, the algorithm returns any valid coloring.

    If the graph is not bipartite, the function returns the value ``False`` and an empty array.

    The input graph can either be:

       - an :class:`~higra.UndirectedGraph` or
       - a triplet of two arrays and an integer (sources, targets, num_vertices)
         defining all the edges of the graph and its number of vertices

    :Examples:

    Example with an undirected graph:

        >>> g = hg.UndirectedGraph(6)
        >>> g.add_edges([0, 0, 4, 2], [1, 4, 3, 3])
        >>> hg.is_bipartite_graph(g)
        (True, array([0, 1, 1, 0, 1, 0]))

    Example with a list of edges:

        >>> hg.is_bipartite_graph(([0, 0, 1, 1, 2, 1, 5], [3, 4, 3, 5, 5, 4, 4], 6))
        (False, array([]))

    :Complexity:

    If the graph is provided as an undirected graph, the algorithm is implemented using a depth first search.
    Its runtime complexity is :math:`O(|V| + |E|)` where :math:`|V|` is the number of vertices and :math:`|E|`
    the number of edges of the graph.

    If the graph is provided as a list of edges (sources, targets, num_vertices), the algorithm is implemented
    using a union find approach. Its runtime complexity is :math:`O(|E|\\times \\alpha(|V|))` time,
    where :math:`\\alpha` is the inverse of the single-valued Ackermann function.

    :param graph: an undirected graph or a triplet of two arrays and an integer (sources, targets, num_vertices)
    :param return_coloring: if True returns a pair (is_bipartite, coloring), otherwise returns only is_bipartite where
           is_bipartite is a boolean and coloring is a 1D array indicating the color of each vertex
    :return: (is_bipartite, coloring) or is_bipartite
    """
    if type(graph) is hg.UndirectedGraph:
        res, coloring = hg.cpp._is_bipartite_graph(graph)
    else:
        try:
            sources = graph[0]
            targets = graph[1]
            num_vertices = graph[2]
            res, coloring = hg.cpp._is_bipartite_graph(sources, targets, num_vertices)
        except Exception as e:
            raise TypeError("graph must be an undirected graph or a tuple (sources, targets, num_vertices)", e)

    if return_coloring:
        return res, coloring
    return res


def bipartite_graph_matching(graph, edge_weights):
    """
    The bipartite graph matching problem is to find a maximum cardinality matching with minimum weight in a bipartite graph.

    This function returns the list of edge indices of the matching.

    The input graph can either be:

       - an :class:`~higra.UndirectedGraph` or
       - a triplet of two arrays and an integer (sources, targets, num_vertices)
         defining all the edges of the graph and its number of vertices

    The input graph must be a bipartite graph.

    Edge weights should be of integral type, otherwise they are casted to int64,
    a warning is emitted, and a possible loss of precision may occur.

    This function is implemented using the CSA library by Andrew V. Goldberg and Robert Kennedy see `https://github.com/rick/CSA <https://github.com/rick/CSA>`_ .
    It is based on a push-relabel method.

    :Examples:

    Example with an undirected graph:

        >>> g = hg.UndirectedGraph(6)
        >>> g.add_edges([5, 5, 1, 1, 4, 6], [2, 3, 2, 3, 0, 0])
        >>> weights = [1, 2, 2, 5, 8, 5]
        >>> hg.bipartite_graph_matching(g, weights)
        array([1, 2, 5])

    Example with a list of edges:

        >>> hg.bipartite_graph_matching(([5, 5, 1, 1, 4, 6], [2, 3, 2, 3, 0, 0], 7), [1, 2, 2, 5, 8, 5])
        array([1, 2, 5])

    :param graph: an undirected graph or a triplet of two arrays and an integer (sources, targets, num_vertices), must be bipartite
    :param edge_weights: edge weights of the graph
    :return: the list of edge indices of the matching
    """

    check, coloring = is_bipartite_graph(graph)

    if type(graph) is hg.UndirectedGraph:
        sources, targets = graph.edge_list()
        num_vertices = graph.num_vertices()
        num_edges = graph.num_edges()
    else:
        try:
            sources = graph[0]
            targets = graph[1]
            num_vertices = graph[2]
            num_edges = len(sources)

            if len(targets) != num_edges:
                raise ValueError("sources and targets must have the same length")

        except Exception as e:
            raise TypeError("graph must be an undirected graph or a tuple (sources, targets, num_vertices)", e)

    if len(edge_weights) != num_edges:
        raise ValueError("edge_weights must have the same length as the number of edges of the graph")

    if not check:
        raise ValueError("graph must be bipartite")

    if num_edges == 0:
        return np.array([], dtype=np.int64)

    # if edge weights dtype is floating point, emit a warning
    if np.issubdtype(edge_weights.dtype, np.floating):
        print("Warning: possible loss of precision, edge weights are casted to int64 for bipartite graph matching")

    edge_weights = hg.cast_to_dtype(edge_weights, np.int64)

    very_large_weight = np.max(edge_weights) * 1000
    # check that very_large_weight is not too large
    if very_large_weight > 2 ** 60 - 1:
        raise ValueError("edge weights are too large")


    # sizes of left and right sets
    rhs_size = np.count_nonzero(coloring)
    lhs_size = num_vertices - rhs_size

    # augmented graph sizes to ensure the existence of a perfect matching
    # each lhs (resp rhs) vertex is duplicated in the rhs (reps lhs) part
    # an edge of weight very_large_weight is added between the two copies
    # each edge of the original graph is duplicated between the mirrored vertices with the same weight

    lhs_augmented_size = rhs_size + lhs_size
    rhs_augmented_size = rhs_size + lhs_size
    num_augmented_vertices = lhs_augmented_size + rhs_augmented_size
    augmented_edge_count = num_edges * 2 + rhs_size + lhs_size
    rhs_augmented_start_id = lhs_augmented_size

    # new vertex ids so that left vertices are first and right vertices are last
    new_vertex_ids = np.zeros(num_vertices, dtype=np.int64)
    new_vertex_ids[coloring == 0] = np.arange(lhs_size)
    # rhs vertices ids start at lhs_size + rhs_size to leave space for lhs vertices augmentation
    new_vertex_ids[coloring == 1] = np.arange(rhs_augmented_start_id, rhs_augmented_start_id + rhs_size)

    augmented_sources = np.zeros(augmented_edge_count, dtype=np.int64)
    augmented_targets = np.zeros(augmented_edge_count, dtype=np.int64)
    augmented_weights = np.zeros(augmented_edge_count, dtype=edge_weights.dtype)

    new_sources = new_vertex_ids[sources]
    new_targets = new_vertex_ids[targets]
    new_sources_ids = np.minimum(new_sources, new_targets)
    new_targets_ids = np.maximum(new_sources, new_targets)

    augmented_sources[:num_edges] = new_sources_ids
    augmented_targets[:num_edges] = new_targets_ids
    augmented_weights[:num_edges] = edge_weights

    # add edges from lhs vertices to rhs vertices with very large weight
    augmented_sources[num_edges:num_edges + lhs_size] = np.arange(lhs_size)
    augmented_targets[num_edges:num_edges + lhs_size] = np.arange(rhs_augmented_start_id + rhs_size, rhs_augmented_start_id + rhs_size + lhs_size)
    augmented_weights[num_edges:num_edges + lhs_size] = very_large_weight

    # add edges from rhs vertices to lhs vertices with very large weight
    augmented_sources[num_edges + lhs_size:num_edges + lhs_size + rhs_size] = np.arange(lhs_size, lhs_size + rhs_size)
    augmented_targets[num_edges + lhs_size:num_edges + lhs_size + rhs_size] = np.arange(lhs_augmented_size, lhs_augmented_size + rhs_size)
    augmented_weights[num_edges + lhs_size:num_edges + lhs_size + rhs_size] = very_large_weight

    # add duplicate of original edges between original lhs and rhs mirrors
    augmented_sources[num_edges + lhs_size + rhs_size:num_edges + lhs_size + rhs_size + num_edges] = new_targets_ids - lhs_augmented_size + lhs_size
    augmented_targets[num_edges + lhs_size + rhs_size:num_edges + lhs_size + rhs_size + num_edges] = new_sources_ids + lhs_augmented_size + rhs_size
    augmented_weights[num_edges + lhs_size + rhs_size:num_edges + lhs_size + rhs_size + num_edges] = edge_weights

    matched_edges = hg.cpp._bipartite_graph_matching(augmented_sources, augmented_targets, num_augmented_vertices, augmented_weights)

    matched_edges = matched_edges[matched_edges < num_edges]

    return matched_edges






