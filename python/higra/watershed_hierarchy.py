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


@hg.data_consumer("edge_weights", "vertex_area")
def watershed_hierarchy_by_area(graph, edge_weights, vertex_area):
    res = hg._watershed_hierarchy_by_area(graph, edge_weights, vertex_area)
    tree = res.tree()
    altitudes = res.node_altitude()

    hg.set_attribute(tree, "leaf_graph", graph)
    hg.set_attribute(tree, "altitudes", altitudes)

    return tree


@hg.data_consumer("edge_weights", "vertex_area")
def watershed_hierarchy_by_volume(graph, edge_weights, vertex_area):
    res = hg._watershed_hierarchy_by_volume(graph, edge_weights, vertex_area)
    tree = res.tree()
    altitudes = res.node_altitude()

    hg.set_attribute(tree, "leaf_graph", graph)
    hg.set_attribute(tree, "altitudes", altitudes)

    return tree


@hg.data_consumer("edge_weights")
def watershed_hierarchy_by_dynamics(graph, edge_weights):
    res = hg._watershed_hierarchy_by_dynamics(graph, edge_weights)
    tree = res.tree()
    altitudes = res.node_altitude()

    hg.set_attribute(tree, "leaf_graph", graph)
    hg.set_attribute(tree, "altitudes", altitudes)

    return tree


@hg.data_consumer("edge_weights")
def watershed_hierarchy_by_attribute(graph, attribute_functor, edge_weights):
    """
    Compute the watershed hierarchy by a user defined attributes.

    The attribute functor is a function that takes a binary partition tree as argument and returns an array with
    the node attribute values for the given tree. The tree will be equipped with the following attributes:
    "altitudes" and "leaf_graph".

    Example
    -------

    Calling watershed_hierarchy_by_area is equivalent to:

        tree = watershed_hierarchy_by_attribute(graph, attribute_area)

    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :param attribute_functor:
    :return: a tree with attributes "altitudes" and "leaf_graph"
    """

    def helper_functor(tree, altitudes):
        hg.set_attribute(tree, "leaf_graph", graph)
        hg.set_attribute(tree, "altitudes", altitudes)

        return attribute_functor(tree)

    res = hg._watershed_hierarchy_by_attribute(graph, edge_weights, helper_functor)
    tree = res.tree()
    altitudes = res.node_altitude()

    hg.set_attribute(tree, "leaf_graph", graph)
    hg.set_attribute(tree, "altitudes", altitudes)

    return tree
