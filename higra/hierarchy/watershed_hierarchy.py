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


@hg.argument_helper("vertex_area")
def watershed_hierarchy_by_area(graph, edge_weights, vertex_area):
    """
    Watershed hierarchy by area.

    The definition of hierarchical watershed follows the one given in:

        J. Cousty, L. Najman.
        `Incremental algorithm for hierarchical minimum spanning forests and saliency of watershed cuts <https://hal-upec-upem.archives-ouvertes.fr/hal-00622505/document>`_.
        ISMM 2011: 272-283.

    The algorithm used is described in:

        Laurent Najman, Jean Cousty, Benjamin Perret.
        `Playing with Kruskal: Algorithms for Morphological Trees in Edge-Weighted Graphs <https://hal.archives-ouvertes.fr/file/index/docid/798621/filename/ismm2013-algo.pdf>`_.
        ISMM 2013: 135-146.

    :param graph: input graph
    :param edge_weights: input graph edge weights
    :param vertex_area: area of the input graph vertices (provided by :func:`~higra.attribute_vertex_area`)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """
    vertex_area = hg.linearize_vertex_weights(vertex_area, graph)

    vertex_area = hg.cast_to_dtype(vertex_area, np.float64)
    res = hg.cpp._watershed_hierarchy_by_area(graph, edge_weights, vertex_area)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


@hg.argument_helper("vertex_area")
def watershed_hierarchy_by_volume(graph, edge_weights, vertex_area):
    """
    Watershed hierarchy by volume.

    The definition of hierarchical watershed follows the one given in:

        J. Cousty, L. Najman.
        `Incremental algorithm for hierarchical minimum spanning forests and saliency of watershed cuts <https://hal-upec-upem.archives-ouvertes.fr/hal-00622505/document>`_.
        ISMM 2011: 272-283.

    The algorithm used is described in:

        Laurent Najman, Jean Cousty, Benjamin Perret.
        `Playing with Kruskal: Algorithms for Morphological Trees in Edge-Weighted Graphs <https://hal.archives-ouvertes.fr/file/index/docid/798621/filename/ismm2013-algo.pdf>`_.
        ISMM 2013: 135-146.

    :param graph: input graph
    :param edge_weights: input graph edge weights
    :param vertex_area: area of the input graph vertices (provided by :func:`~higra.attribute_vertex_area`)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """
    vertex_area = hg.linearize_vertex_weights(vertex_area, graph)

    vertex_area = hg.cast_to_dtype(vertex_area, np.float64)
    res = hg.cpp._watershed_hierarchy_by_volume(graph, edge_weights, vertex_area)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def watershed_hierarchy_by_dynamics(graph, edge_weights):
    """
    Watershed hierarchy by dynamics.

    The definition of hierarchical watershed follows the one given in:

        J. Cousty, L. Najman.
        `Incremental algorithm for hierarchical minimum spanning forests and saliency of watershed cuts <https://hal-upec-upem.archives-ouvertes.fr/hal-00622505/document>`_.
        ISMM 2011: 272-283.

    The algorithm used is described in:

        Laurent Najman, Jean Cousty, Benjamin Perret.
        `Playing with Kruskal: Algorithms for Morphological Trees in Edge-Weighted Graphs <https://hal.archives-ouvertes.fr/file/index/docid/798621/filename/ismm2013-algo.pdf>`_.
        ISMM 2013: 135-146.

    :param graph: input graph
    :param edge_weights: input graph edge weights
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """
    res = hg.cpp._watershed_hierarchy_by_dynamics(graph, edge_weights)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def watershed_hierarchy_by_number_of_parents(graph, edge_weights):
    """
    Watershed hierarchy by number of parents.

    The definition of *number of parents* was proposed in:

        B. Perret, J. Cousty, S. J. F. Guimarães and D. S. Maia,
        `Evaluation of Hierarchical Watersheds <https://hal.archives-ouvertes.fr/hal-01430865/file/PCGM%20-%20TIP%202018%20-%20Evaluation%20of%20hierarchical%20watersheds.pdf>`_ ,
        in IEEE Transactions on Image Processing, vol. 27, no. 4, pp. 1676-1688, April 2018.
        doi: 10.1109/TIP.2017.2779604


    The definition of hierarchical watershed follows the one given in:

        J. Cousty, L. Najman.
        `Incremental algorithm for hierarchical minimum spanning forests and saliency of watershed cuts <https://hal-upec-upem.archives-ouvertes.fr/hal-00622505/document>`_.
        ISMM 2011: 272-283.

    The algorithm used is described in:

        Laurent Najman, Jean Cousty, Benjamin Perret.
        `Playing with Kruskal: Algorithms for Morphological Trees in Edge-Weighted Graphs <https://hal.archives-ouvertes.fr/file/index/docid/798621/filename/ismm2013-algo.pdf>`_.
        ISMM 2013: 135-146.


    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    def num_parents(bpt_tree, altitudes):
        # construct quasi flat zone hierarchy from input bpt
        tree, node_map = hg.simplify_tree(bpt_tree, altitudes == altitudes[bpt_tree.parents()])

        # determine inner nodes of the min tree, i.e. nodes of qfz having at least one node that is not a leaf
        num_children = tree.num_children(np.arange(tree.num_vertices()))
        num_children_leaf = np.zeros((tree.num_vertices(),), dtype=np.int64)
        np.add.at(num_children_leaf, tree.parents()[:tree.num_leaves()], 1)
        inner_nodes = num_children != num_children_leaf

        # go back into bpt space
        inner_nodes_bpt = np.zeros((bpt_tree.num_vertices(),), dtype=np.int64)
        inner_nodes_bpt[node_map] = inner_nodes
        inner_nodes = inner_nodes_bpt

        # count number of min tree inner nodes in the subtree rooted in the given node
        res = hg.accumulate_and_add_sequential(bpt_tree,
                                               inner_nodes,
                                               inner_nodes[:tree.num_leaves()],
                                               hg.Accumulators.sum)

        # add 1 to avoid having a zero measure in a minima
        res[bpt_tree.num_leaves():] = res[bpt_tree.num_leaves():] + 1

        return res

    return hg.watershed_hierarchy_by_attribute(graph, edge_weights, num_parents)


def watershed_hierarchy_by_attribute(graph, edge_weights, attribute_functor):
    """
    Watershed hierarchy by a user defined attributes.

    The definition of hierarchical watershed follows the one given in:

        J. Cousty, L. Najman.
        `Incremental algorithm for hierarchical minimum spanning forests and saliency of watershed cuts <https://hal-upec-upem.archives-ouvertes.fr/hal-00622505/document>`_.
        ISMM 2011: 272-283.

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

    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :param attribute_functor: function computing the regional attribute
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    def helper_functor(tree, altitudes):
        hg.CptHierarchy.link(tree, graph)

        return attribute_functor(tree, altitudes)

    res = hg.cpp._watershed_hierarchy_by_attribute(graph, edge_weights, helper_functor)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def watershed_hierarchy_by_minima_ordering(graph, edge_weights, minima_ranks, minima_altitudes):
    """
    Watershed hierarchy for the given minima ordering.

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
     
     
    The ranking ranking of the minima of the given edge weighted graph :math:`(G,w)` is given as vertex weights with values in
    :math:`\{0, \ldots, n\}` with :math:`n` the number of minima of :math:`(G,w)`. It must satisfy the following pre-conditions:

        - each minimum of :math:`(G,w)` contains at least one non zero vertex,
        - all non zero vertices in a minimum have the same weight,
        - there is no non zero value vertex outside minima, and
        - no two minima contain non zero vertices with the same weight.
     
    The altitude associated to each minimum is a non decreasing 1d array of size :math:`n + 1` with non negative values.
    Note that the first entry of the minima altitudes array, ie. the value at index 0, does not represent a minimum and
    its value should be 0.

    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :param minima_ranks: input graph vertex weights containing the rank of each minima of the input edge weighted graph
    :param minima_altitudes: array mapping each minima rank to its altitude
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    minima_ranks = hg.cast_to_dtype(minima_ranks, np.uint64)
    minima_altitudes = hg.cast_to_dtype(minima_altitudes, np.float64)
    res = hg.cpp._watershed_hierarchy_by_minima_ordering(graph, edge_weights, minima_ranks, minima_altitudes)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes
