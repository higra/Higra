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


@hg.data_consumer("edge_weights")
def bpt_canonical(graph, edge_weights):
    """
    Compute the canonical binary partition tree (binary tree by altitude ordering) of the given weighted graph.


    :param graph:
    :param edge_weights:
    :return: Tree (with attributes "leaf_graph", "altitudes" and "mst")
    """

    res = hg._bpt_canonical(graph, edge_weights)
    tree = res.tree()
    altitudes = res.altitudes()
    mst = res.mst()

    original_graph = hg.get_attribute(graph, "original_graph")
    if original_graph:
        hg.set_attribute(tree, "leaf_graph", original_graph)
        hg.set_attribute(mst, "original_graph", original_graph)
    else:
        hg.set_attribute(tree, "leaf_graph", graph)
        hg.set_attribute(mst, "original_graph", graph)

    hg.set_attribute(tree, "altitudes", altitudes)
    hg.set_attribute(tree, "mst", mst)

    return tree

@hg.data_consumer("edge_weights")
def quasi_flat_zones_hierarchy(graph, edge_weights):
    """
    Compute the quasi flat zones hierarchy of the given weighted graph.


    :param graph:
    :param edge_weights:
    :return: Tree (with attributes "leaf_graph" and "altitudes")
    """

    res = hg._quasi_flat_zones_hierarchy(graph, edge_weights)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.set_attribute(tree, "leaf_graph", graph)
    hg.set_attribute(tree, "altitudes", altitudes)

    return tree


@hg.data_consumer("deleted_vertices")
def simplify_tree(tree, deleted_vertices):
    """
    Creates a copy of the current Tree and deletes the vertices i such that deletedVertices[i] is true.

    The attribute "node_map" of the returned tree is an array that maps any node index i of the new tree,
    to the index of this node in the original tree.

    :param tree:
    :param deleted_vertices:
    :return: simplified tree (with attributes "node_map")
    """

    res = hg._simplify_tree(tree, deleted_vertices)
    new_tree = res.tree()
    node_map = res.node_map()

    hg.set_attribute(new_tree, "leaf_graph", hg.getAttribute(tree, "leaf_graph"))
    hg.set_attribute(new_tree, "node_map", node_map)

    return new_tree


@hg.data_consumer("altitudes", "lca_map")
def saliency(tree, altitudes, lca_map, propagate_if_rag=True):
    """
    Compute the saliency map (ultra-metric distance) of the given tree
    :param tree:
    :param altitudes: altitudes of the vertices of the tree
    :param lca_map: array containing the lowest common ancestor of the source and target vertices of each edge
    where saliency need to be computed
    :param propagate_if_rag: if tree has been constructed on a rag, then saliency values will be propagated to the
    original graph, hence leading to a saliency on the original graph and not on the rag
    :return: edge saliency corresponding to the given tree
    """
    sm = altitudes[lca_map]
    graph = hg.get_attribute(tree, "leaf_graph")
    if hg.get_attribute(graph, "pre_graph") and propagate_if_rag:
        sm = hg.rag_back_project_edge_weights(graph, sm)
    return sm


@hg.data_consumer("edge_weights")
def binary_partition_tree_complete_linkage(graph, edge_weights):
    """
    Compute a binary partition tree with complete linkage distance.

    Given a graph G, with initial edge weights W,
    the distance d(X,Y) between any two regions X, Y is defined as :\n
    d(X,Y) = max {W({x,y}) | x in X, y in Y, {x,y} in G }

    :param graph:
    :param edge_weights:
    :return:
    """

    res = hg._binary_partition_tree_complete_linkage(graph, edge_weights)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.set_attribute(tree, "leaf_graph", graph)
    hg.set_attribute(tree, "altitudes", altitudes)

    return tree


def binary_partition_tree_average_linkage(graph, edge_values, edge_weights):
    """
    Compute a binary partition tree with average linkage distance.

    Given a graph G, with initial edge values V with associated weights W,
    the distance d(X,Y) between any two regions X, Y is defined as:
    d(X,Y) = (1 / Z) + sum_{x in X, y in Y, {x,y} in G} V({x,y}) x W({x,y})
    with Z = sum_{x in X, y in Y, {x,y} in G} W({x,y})

    :param graph:
    :param edge_values:
    :param edge_weights:
    :return:
    """

    res = hg._binary_partition_tree_average_linkage(graph, edge_values,  edge_weights)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.set_attribute(tree, "leaf_graph", graph)
    hg.set_attribute(tree, "altitudes", altitudes)

    return tree


@hg.data_consumer("edge_weights")
def binary_partition_tree_single_linkage(graph, edge_weights):
    """
    Alias for bpt_canonical

    :param graph:
    :param edge_weights:
    :return:
    """

    return bpt_canonical(graph, edge_weights)


@hg.data_consumer("edge_weights")
def binary_partition_tree(graph, weight_function, edge_weights):
    """
    Compute the binary partition tree of the graph.
     
    At each step:
    1 - the algorithm finds the edge of smallest weight.
    2 - the two vertices linked by this edge are merged: the new vertex is the parent of the two merged vertices
    3 - the weight of the edges linking the new vertex to the remaining vertices of the graph are updated according
       to the user provided function (weight_function)
    4 - repeat until a single edge remain

    The initial weight of the edges (edge_weights) and the callback (weight_function) determine the shape of the
    hierarchy.

    Classical single/complete/average linkage weighting function are already implemented: use functions
    - binary_partition_tree_single_linkage
    - binary_partition_tree_complete_linkage
    - binary_partition_tree_average_linkage.
    Those functions are impleted and c++ and should be faster than a user defined weighting function written in Python.

    The weight_function callback can be anything that defining the operator() and should follow the following pattern:

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
    - num_edges(): returns 2 if both the two merged vertices add an edge linking themselves with neighbour_vertex()
       and 1 otherwise
    - first_edge_index(): the index of the edge linking one of the merged region to neighbour_vertex()
    - second_edge_index(): the index of the edge linking the other merged region to neighbour_vertex() (only if num_edges()==2)
    - set_new_edge_weight(value): weight of the new edge (THIS HAS TO BE DEFINED IN THE WEIGHTING FUNCTION)
    - new_edge_index(): the index of the new edge: the weighting function will probably have to track new weight values

    Example of weighting function for average linkage assuming that
    - edge_weights is an array containing the weight of each edge, and
    - edge_values is an array containing the value of each edge:

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

    :param graph:
    :param edge_weights:
    :param weight_function:
    :return:
    """
    res = hg._binary_partition_tree_custom_linking(graph, edge_weights, weight_function)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.set_attribute(tree, "leaf_graph", graph)
    hg.set_attribute(tree, "altitudes", altitudes)

    return tree