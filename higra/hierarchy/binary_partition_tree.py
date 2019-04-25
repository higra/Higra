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

@hg.argument_helper(hg.CptEdgeWeightedGraph)
def binary_partition_tree_complete_linkage(edge_weights, graph):
    """
    Compute a binary partition tree with complete linkage distance.

    Given a graph G, with initial edge weights W,
    the distance d(X,Y) between any two regions X, Y is defined as :\n
    d(X,Y) = max {W({x,y}) | x in X, y in Y, {x,y} in G }

    :param edge_weights: edge weights of the input graph (Concept :class:`~higra.CptEdgeWeightedGraph`)
    :param graph: input graph (deduced from :class:`~higra.CptEdgeWeightedGraph`)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes (Concept :class:`~higra.CptValuedHierarchy`)
    """

    res = hg.cpp._binary_partition_tree_complete_linkage(graph, edge_weights)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)
    hg.CptValuedHierarchy.link(altitudes, tree)

    return tree, altitudes


@hg.argument_helper(hg.CptEdgeWeightedGraph)
def binary_partition_tree_average_linkage(edge_weights, graph, edge_weight_weights=None):
    """
    Compute a binary partition tree with average linkage distance.

    Given a graph G, with initial edge weights V with associated weighting W,
    the distance d(X,Y) between any two regions X, Y is defined as:
    d(X,Y) = (1 / Z) + sum_{x in X, y in Y, {x,y} in G} V({x,y}) x W({x,y})
    with Z = sum_{x in X, y in Y, {x,y} in G} W({x,y})

    :param edge_weights: edge weights of the input graph (Concept :class:`~higra.CptEdgeWeightedGraph`)
    :param graph: input graph (deduced from :class:`~higra.CptEdgeWeightedGraph`)
    :param edge_weight_weights: weighting of edge weights of the input graph (default to an array of ones)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes (Concept :class:`~higra.CptValuedHierarchy`)
    """

    if edge_weight_weights is None:
        edge_weight_weights = np.ones_like(edge_weights)
    else:
        edge_weights, edge_weight_weights = hg.cast_to_common_type(edge_weights, edge_weight_weights)

    res = hg.cpp._binary_partition_tree_average_linkage(graph, edge_weights, edge_weight_weights)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)
    hg.CptValuedHierarchy.link(altitudes, tree)

    return tree, altitudes


@hg.argument_helper(hg.CptEdgeWeightedGraph)
def binary_partition_tree_single_linkage(edge_weights, graph):
    """
    Alias for bpt_canonical

    :param edge_weights: edge weights of the input graph (Concept :class:`~higra.CptEdgeWeightedGraph`)
    :param graph: input graph (deduced from :class:`~higra.CptEdgeWeightedGraph`)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes (Concept :class:`~higra.CptValuedHierarchy`)
    """

    return hg.bpt_canonical(edge_weights, graph)


@hg.argument_helper(("edge_weights", hg.CptEdgeWeightedGraph))
def binary_partition_tree(weight_function, edge_weights, graph):
    """
    Compute the binary partition tree of the graph.

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

    :param weight_function: see detailed description above
    :param edge_weights: edge weights of the input graph (Concept :class:`~higra.CptEdgeWeightedGraph`)
    :param graph: input graph (deduced from :class:`~higra.CptEdgeWeightedGraph`)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes (Concept :class:`~higra.CptValuedHierarchy`)
    """
    res = hg.cpp._binary_partition_tree(graph, edge_weights, weight_function)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)
    hg.CptValuedHierarchy.link(altitudes, tree)

    return tree, altitudes
