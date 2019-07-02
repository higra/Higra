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


def binary_partition_tree_complete_linkage(graph, edge_weights):
    """
    Binary partition tree with complete linkage distance.

    Given a graph :math:`G=(V, E)`, with initial edge weights :math:`w`,
    the distance :math:`d(X,Y)` between any two clusters :math:`X` and :math:`Y` is

    .. math::

        d(X,Y) = \max \{w(\{x,y\}) | x \in X, y \in Y, \{x,y\} \in E \}

    Regions are then iteratively merged following the above distance (closest first) until a single region remains

    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    res = hg.cpp._binary_partition_tree_complete_linkage(graph, edge_weights)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def binary_partition_tree_average_linkage(graph, edge_weights, edge_weight_weights=None):
    """
    Binary partition tree with average linkage distance.

    Given a graph :math:`G=(V, E)`, with initial edge weights :math:`w` with associated weights :math:`w_2`,
    the distance :math:`d(X,Y)` between any two clusters :math:`X` and :math:`Y` is

    .. math::

        d(X,Y) = \\frac{1}{Z} \sum_{x \in X, y \in Y, \{x,y\} \in E} w(\{x,y\}) \\times w_2(\{x,y\})

    with :math:`Z = \sum_{x \in X, y \in Y, \{x,y\} \in E} w_2({x,y})`.

    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :param edge_weight_weights: weighting of edge weights of the input graph (default to an array of ones)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    if edge_weight_weights is None:
        edge_weight_weights = np.ones_like(edge_weights)
    else:
        edge_weights, edge_weight_weights = hg.cast_to_common_type(edge_weights, edge_weight_weights)

    res = hg.cpp._binary_partition_tree_average_linkage(graph, edge_weights, edge_weight_weights)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def binary_partition_tree_exponential_linkage(graph, edge_weights, alpha, edge_weight_weights=None):
    """
    Binary partition tree with exponential linkage distance.

    Given a graph :math:`G=(V, E)`, with initial edge weights :math:`w` with associated weights :math:`w_2`,
    the distance :math:`d(X,Y)` between any two clusters :math:`X` and :math:`Y` is

    .. math::

         d(X,Y) = \\frac{1}{Z} \sum_{x \in X, y \in Y, \{x,y\} in E} w_2(\{x,y\}) \\times \exp(\\alpha * w(\{x,y\})) \\times w(\{x,y\})

    with :math:`Z = \sum_{x \in X, y \in Y, \{x,y\} \in E} w_2(\{x,y\}) \\times \exp(\\alpha * w(\{x,y\}))`.

    :See:

         Nishant Yadav, Ari Kobren, Nicholas Monath, Andrew Mccallum.
         `Supervised Hierarchical Clustering with Exponential Linkage <http://proceedings.mlr.press/v97/yadav19a.html>`_
         Proceedings of the 36th International Conference on Machine Learning, PMLR 97:6973-6983, 2019.

    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :param alpha: exponential parameter
    :param edge_weight_weights: weighting of edge weights of the input graph (default to an array of ones)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    alpha = float(alpha)

    if edge_weight_weights is None:
        edge_weight_weights = np.ones_like(edge_weights)
    else:
        edge_weights, edge_weight_weights = hg.cast_to_common_type(edge_weights, edge_weight_weights)

    # special cases: improve efficiency and avoid numerical issues
    if alpha == 0:
        tree, altitudes = hg.binary_partition_tree_average_linkage(graph, edge_weights, edge_weight_weights)
    elif alpha == float('-inf'):
        tree, altitudes = hg.binary_partition_tree_single_linkage(graph, edge_weights)
    elif alpha == float('inf'):
        tree, altitudes = hg.binary_partition_tree_complete_linkage(graph, edge_weights)
    else:
        res = hg.cpp._binary_partition_tree_exponential_linkage(graph, edge_weights, alpha, edge_weight_weights)
        tree = res.tree()
        altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def binary_partition_tree_single_linkage(graph, edge_weights):
    """
    Alias for :func:`~higra.bpt_canonical`.

    Given a graph :math:`G=(V, E)`, with initial edge weights :math:`w`,
    the distance :math:`d(X,Y)` between any two clusters :math:`X` and :math:`Y` is

    .. math::

        d(X,Y) = \min \{w(\{x,y\}) | x \in X, y \in Y, \{x,y\} \in E \}

    Regions are then iteratively merged following the above distance (closest first) until a single region remains.

    :param edge_weights: edge weights of the input graph (Concept :class:`~higra.CptEdgeWeightedGraph`)
    :param graph: input graph (deduced from :class:`~higra.CptEdgeWeightedGraph`)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes (Concept :class:`~higra.CptValuedHierarchy`)
    """

    return hg.bpt_canonical(graph, edge_weights)


def binary_partition_tree_ward_linkage(graph, vertex_centroids, vertex_sizes=None, altitude_correction="max"):
    """
    Binary partition tree with the Ward linkage rule.

    Given a graph :math:`G=(V, E)`, with initial edge weights :math:`w` with associated weights :math:`w'`,
    the distance :math:`d(X,Y)` between any two clusters :math:`X` and :math:`Y` is

    .. math::

        d(X,Y) = \\frac{| X |\\times| Y |}{| X |+| Y |} \| \\vec{X} - \\vec{Y} \|^2

    where :math:`\\vec{X}` and :math:`\\vec{Y}` are the centroids of  :math:`X` and  :math:`Y`.

    Regions are then iteratively merged following the above distance (closest first) until a single region remains

    Note that the Ward distance is not necessarily strictly increasing when processing a non complete graph.
    This can be corrected afterward with an altitude correction strategy. Valid values for ``altitude correction`` are:

        - ``"none"``: nothing is done and the altitude of a node is equal to the Ward distance between its 2 children;
          this may not be non-decreasing
        - ``"max"``: the altitude of a node :math:`n` is defined as the maximum of the the Ward distance associated
          to each node in the subtree rooted in :math:`n`.

    :param graph: input graph
    :param vertex_centroids: Centroids of the graph vertices (must be a 2d array)
    :param vertex_sizes: Size (number of elements) of the graph vertices (default to an array of ones)
    :param altitude_correction: can be ``"none"`` or ``"max"`` (default)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    if vertex_sizes is None:
        vertex_sizes = np.ones((graph.num_vertices(),), dtype=vertex_centroids.dtype)
    else:
        vertex_centroids, vertex_sizes = hg.cast_to_common_type(vertex_centroids, vertex_sizes)

    res = hg.cpp._binary_partition_tree_ward_linkage(graph, vertex_centroids, vertex_sizes, altitude_correction)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def binary_partition_tree(graph, weight_function, edge_weights):
    """
    Binary partition tree of the graph with a user provided cluster distance.

    At each step:

    1 - the algorithm finds the edge of smallest weight.
    2 - the two vertices linked by this edge are merged: the new vertex is the parent of the two merged vertices
    3 - the weight of the edges linking the new vertex to the remaining vertices of the graph are updated according to the user provided function (weight_function)
    4 - repeat until a single edge remain

    The initial weight of the edges (edge_weights) and the callback (weight_function) determine the shape of the
    hierarchy.

    Classical single/complete/average linkage weighting function are already implemented: use functions

    - binary_partition_tree_single_linkage
    - binary_partition_tree_complete_linkage
    - binary_partition_tree_average_linkage.

    Those functions are implemented in c++ and should be faster than a user defined weighting function written in Python.

    The weight_function callback can be anything that defining the operator() and should follow the following pattern:

    .. code-block:: python

        def weight_function(graph,              # the current state of the graph
                       fusion_edge_index,       # the edge between the two vertices being merged
                       new_region,              # the new vertex in the graph
                       merged_region1,          # the first vertex merged
                       merged_region2,          # the second vertex merged
                       new_neighbours):         # list of edges to be weighted (see below)
           ...
           for n in new_neighbours:
               ...
               n.set_new_edge_weight(new_edge_value) # define the weight of this edge

        }

    Each element in the parameter new_neighbours represent an edge between the new vertex and another vertex of
    the graph. For each element of the list, the following methods are available:

    - neighbour_vertex(): the other vertex
    - num_edges(): returns 2 if both the two merged vertices add an edge linking themselves with neighbour_vertex() and 1 otherwise
    - first_edge_index(): the index of the edge linking one of the merged region to neighbour_vertex()
    - second_edge_index(): the index of the edge linking the other merged region to neighbour_vertex() (only if num_edges()==2)
    - set_new_edge_weight(value): weight of the new edge (THIS HAS TO BE DEFINED IN THE WEIGHTING FUNCTION)
    - new_edge_index(): the index of the new edge: the weighting function will probably have to track new weight values

    Example of weighting function for average linkage assuming that

    - edge_weights is an array containing the weight of each edge, and
    - edge_values is an array containing the value of each edge:

    .. code-block:: python

        def weighting_function_average_linkage(graph, fusion_edge_index, new_region, merged_region1, merged_region2, new_neighbours):
            for n in new_neighbours:
                if n.num_edges() > 1:
                    new_weight = edge_weights[n.first_edge_index()] + edge_weights[n.second_edge_index()]
                    new_value = (edge_values[n.first_edge_index()] * edge_weights[n.first_edge_index()] \
                        + edge_values[n.second_edge_index()] * edge_weights[n.second_edge_index()]) \
                        / new_weight
                else:
                    new_weight = edge_weights[n.first_edge_index()]
                    new_value = edge_values[n.first_edge_index()]

                n.set_new_edge_weight(new_value)
                edge_values[n.new_edge_index()] = new_value
                edge_weights[n.new_edge_index()] = new_weight

    :param graph: input graph
    :param weight_function: see detailed description above
    :param edge_weights: edge weights of the input graph
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """
    res = hg.cpp._binary_partition_tree(graph, edge_weights, weight_function)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes
