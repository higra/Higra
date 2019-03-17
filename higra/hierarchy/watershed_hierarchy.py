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

    The algorithm used is described in:

        Laurent Najman, Jean Cousty, Benjamin Perret.
        `Playing with Kruskal: Algorithms for Morphological Trees in Edge-Weighted Graphs <https://hal.archives-ouvertes.fr/file/index/docid/798621/filename/ismm2013-algo.pdf>`_.
        ISMM 2013: 135-146.

    The attribute functor is a function that takes a binary partition tree and an array of altitudes as argument
    and returns an array with the node attribute values for the given tree.

    Example:

    Calling watershed_hierarchy_by_area is equivalent to:

    .. code-block:: python

        tree = watershed_hierarchy_by_attribute(graph, lambda tree, _: hg.attribute_area(tree))


    :param edge_weights: edge weights of the input graph (Concept :class:`~higra.CptEdgeWeightedGraph`)
    :param attribute_functor: function computing the regional attribute
    :param graph: input graph (deduced from :class:`~higra.CptEdgeWeightedGraph`)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes (Concept :class:`~higra.CptValuedHierarchy`)
    """

    def helper_functor(tree, altitudes):
        hg.CptHierarchy.link(tree, graph)
        hg.CptValuedHierarchy.link(altitudes, tree)

        return attribute_functor(tree, altitudes)

    res = hg.cpp._watershed_hierarchy_by_attribute(graph, edge_weights, helper_functor)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)
    hg.CptValuedHierarchy.link(altitudes, tree)

    return tree, altitudes


@hg.argument_helper(hg.CptEdgeWeightedGraph)
def watershed_hierarchy_by_minima_ordering(edge_weights, minima_ranks, minima_altitudes, graph):
    """
    Computes a hierarchical watershed for the given minima ordering.

    The definition used follows the one given in:

        J. Cousty, L. Najman.
        `Incremental algorithm for hierarchical minimum spanning forests and saliency of watershed cuts <https://hal-upec-upem.archives-ouvertes.fr/hal-00622505/document>`_.
        ISMM 2011: 272-283.

    and in,

        J. Cousty, L. Najman, B. Perret.
        `Constructive links between some morphological hierarchies on edge-weighted graphs <https://hal.archives-ouvertes.fr/file/index/docid/806851/filename/ismm2013.pdf>`_..
        ISMM 2013: 86-97.
     
    The algorithm used is adapted from the algorithm described in:
     
        Laurent Najman, Jean Cousty, Benjamin Perret.
        `Playing with Kruskal: Algorithms for Morphological Trees in Edge-Weighted Graphs <https://hal.archives-ouvertes.fr/file/index/docid/798621/filename/ismm2013-algo.pdf>`_.
        ISMM 2013: 135-146.
     
     
    The ranking ranking of the minima of the given edge weighted graph (G,w) is given as vertex weights with values in
    {0..n} with n the number of minima of (G,w). It must satisfy the following pre-conditions:

        - each minimum of (G,w) contains at least one non zero vertex,
        - all non zero vertices in a minimum have the same weight,
        - there is no non zero value vertex outside minima, and
        - no two minima contain non zero vertices with the same weight.
     
    The altitude associated to each minimum is a non decreasing 1d array of size n + 1 with non negative values.
    Note that the first entry of the minima altitudes array, ie. the value at index 0, does not represent a minimum and
    its value should be 0.

    :param edge_weights: edge weights of the input graph (Concept :class:`~higra.CptEdgeWeightedGraph`)
    :param minima_ranks: input graph vertex weights containing the rank of each minima of the input edge weighted graph
    :param minima_altitudes: array mapping each minima rank to its altitude
    :param graph: input graph (deduced from :class:`~higra.CptEdgeWeightedGraph`)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes (Concept :class:`~higra.CptValuedHierarchy`)
    """
    res = hg.cpp._watershed_hierarchy_by_minima_ordering(graph, edge_weights, minima_ranks, minima_altitudes)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)
    hg.CptValuedHierarchy.link(altitudes, tree)

    return tree, altitudes
