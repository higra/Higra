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