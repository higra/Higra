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


@hg.argument_helper(hg.CptEdgeWeightedGraph, ("graph", "vertex_area"))
def watershed_hierarchy_by_area(edge_weights, graph, vertex_area):
    vertex_area = hg.linearize_vertex_weights(vertex_area, graph)
    res = hg._watershed_hierarchy_by_area(graph, edge_weights, vertex_area)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)
    hg.CptValuedHierarchy.link(altitudes, tree)

    return tree, altitudes


@hg.argument_helper(hg.CptEdgeWeightedGraph, ("graph", "vertex_area"))
def watershed_hierarchy_by_volume(edge_weights, graph, vertex_area):
    vertex_area = hg.linearize_vertex_weights(vertex_area, graph)
    res = hg._watershed_hierarchy_by_volume(graph, edge_weights, vertex_area)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)
    hg.CptValuedHierarchy.link(altitudes, tree)

    return tree, altitudes


@hg.argument_helper(hg.CptEdgeWeightedGraph)
def watershed_hierarchy_by_dynamics(edge_weights, graph):
    res = hg._watershed_hierarchy_by_dynamics(graph, edge_weights)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)
    hg.CptValuedHierarchy.link(altitudes, tree)

    return tree, altitudes


@hg.argument_helper(hg.CptEdgeWeightedGraph)
def watershed_hierarchy_by_attribute(edge_weights, attribute_functor, graph):
    """
    Compute the watershed hierarchy by a user defined attributes.

    The attribute functor is a function that takes a binary partition tree and an array of altitudes as argument
    and returns an array with the node attribute values for the given tree.

    Example:

    Calling watershed_hierarchy_by_area is equivalent to:

    .. code-block:: python

        tree = watershed_hierarchy_by_attribute(graph, lambda tree, _: hg.attribute_area(tree))

    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :param attribute_functor:
    :return: a tree with attributes "altitudes" and "leaf_graph"
    """

    def helper_functor(tree, altitudes):
        hg.CptHierarchy.link(tree, graph)
        hg.CptValuedHierarchy.link(altitudes, tree)

        return attribute_functor(tree, altitudes)

    res = hg._watershed_hierarchy_by_attribute(graph, edge_weights, helper_functor)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)
    hg.CptValuedHierarchy.link(altitudes, tree)

    return tree, altitudes
