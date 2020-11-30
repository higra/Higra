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


def watershed_hierarchy_by_area(graph, edge_weights, vertex_area=None, canonize_tree=True):
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
    :param canonize_tree: if ``True`` (default), the resulting hierarchy is canonized (see function :func:`~higra.canonize_hierarchy`),
           otherwise the returned hierarchy is a binary tree
    :return: a tree (Concept :class:`~higra.CptHierarchy` is ``True`` and :class:`~higra.CptBinaryHierarchy` otherwise)
             and its node altitudes
    """
    if vertex_area is None:
        vertex_area = hg.attribute_vertex_area(graph)

    vertex_area = hg.linearize_vertex_weights(vertex_area, graph)

    return watershed_hierarchy_by_attribute(graph, edge_weights,
                                            lambda tree, _: hg.attribute_area(tree, vertex_area, graph),
                                            canonize_tree)


def watershed_hierarchy_by_volume(graph, edge_weights, vertex_area=None, canonize_tree=True):
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
    :param canonize_tree: if ``True`` (default), the resulting hierarchy is canonized (see function :func:`~higra.canonize_hierarchy`),
           otherwise the returned hierarchy is a binary tree
    :return: a tree (Concept :class:`~higra.CptHierarchy` is ``True`` and :class:`~higra.CptBinaryHierarchy` otherwise)
             and its node altitudes
    """
    if vertex_area is None:
        vertex_area = hg.attribute_vertex_area(graph)

    vertex_area = hg.linearize_vertex_weights(vertex_area, graph)

    return watershed_hierarchy_by_attribute(graph, edge_weights,
                                            lambda tree, altitudes:
                                            hg.attribute_volume(tree,
                                                                altitudes,
                                                                hg.attribute_area(tree, vertex_area)),
                                            canonize_tree)


def watershed_hierarchy_by_dynamics(graph, edge_weights, canonize_tree=True):
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
    :param canonize_tree: if ``True`` (default), the resulting hierarchy is canonized (see function :func:`~higra.canonize_hierarchy`),
           otherwise the returned hierarchy is a binary tree
    :return: a tree (Concept :class:`~higra.CptHierarchy` is ``True`` and :class:`~higra.CptBinaryHierarchy` otherwise)
             and its node altitudes
    """
    return watershed_hierarchy_by_attribute(graph, edge_weights,
                                            lambda tree, altitudes:
                                            hg.attribute_dynamics(tree,
                                                                  altitudes,
                                                                  "increasing"),
                                            canonize_tree)


def watershed_hierarchy_by_number_of_parents(graph, edge_weights, canonize_tree=True):
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
    :param canonize_tree: if ``True`` (default), the resulting hierarchy is canonized (see function :func:`~higra.canonize_hierarchy`),
           otherwise the returned hierarchy is a binary tree
    :return: a tree (Concept :class:`~higra.CptHierarchy` is ``True`` and :class:`~higra.CptBinaryHierarchy` otherwise)
             and its node altitudes
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

    return hg.watershed_hierarchy_by_attribute(graph, edge_weights, num_parents, canonize_tree)


def watershed_hierarchy_by_attribute(graph, edge_weights, attribute_functor, canonize_tree=True):
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

        tree = watershed_hierarchy_by_attribute(graph, edge_weights, lambda tree, _: hg.attribute_area(tree))

    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :param attribute_functor: function computing the regional attribute
    :param canonize_tree: if ``True`` (default), the resulting hierarchy is canonized (see function :func:`~higra.canonize_hierarchy`),
           otherwise the returned hierarchy is a binary tree
    :return: a tree (Concept :class:`~higra.CptHierarchy` is ``True`` and :class:`~higra.CptBinaryHierarchy` otherwise)
             and its node altitudes
    """

    def helper_functor(tree, altitudes):
        hg.CptHierarchy.link(tree, graph)

        return attribute_functor(tree, altitudes)

    tree, altitudes, mst_edge_map = hg.cpp._watershed_hierarchy_by_attribute(graph, edge_weights, helper_functor)

    hg.CptHierarchy.link(tree, graph)

    if canonize_tree:
        tree, altitudes = hg.canonize_hierarchy(tree, altitudes)
    else:
        mst = hg.subgraph(graph, mst_edge_map)
        hg.CptMinimumSpanningTree.link(mst, graph, mst_edge_map)
        hg.CptBinaryHierarchy.link(tree, mst_edge_map, mst)

    return tree, altitudes


def watershed_hierarchy_by_minima_ordering(graph, edge_weights, minima_ranks, minima_altitudes=None, canonize_tree=True):
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
     
    :attr:`minima_altitudes` is an optional non decreasing 1d array of size :math:`n + 1` with non negative values such that
    :math:`minima\_altitudes[i]` indicates the altitude of the minima of rank :math:`i`.
    Note that the first entry of the minima altitudes array, ie. the value at index 0, does not represent a minimum and
    its value should be 0.

    The altitude of a node of the computed watershed corresponds to the altitude (respectively the rank) of the minima it
    is associated to if :attr:`minima_altitudes` is provided (respectively not provided).

    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :param minima_ranks: input graph vertex weights containing the rank of each minima of the input edge weighted graph
    :param minima_altitudes: array mapping each minima rank to its altitude (optional)
    :param canonize_tree: if ``True`` (default), the resulting hierarchy is canonized (see function :func:`~higra.canonize_hierarchy`),
           otherwise the returned hierarchy is a binary tree
    :return: a tree (Concept :class:`~higra.CptHierarchy` is ``True`` and :class:`~higra.CptBinaryHierarchy` otherwise)
             and its node altitudes
    """

    minima_ranks = hg.cast_to_dtype(minima_ranks, np.uint64)
    tree, altitudes, mst_edge_map = hg.cpp._watershed_hierarchy_by_minima_ordering(graph, edge_weights, minima_ranks)

    hg.CptHierarchy.link(tree, graph)

    if canonize_tree:
        tree, altitudes = hg.canonize_hierarchy(tree, altitudes)
    else:
        mst = hg.subgraph(graph, mst_edge_map)
        hg.CptMinimumSpanningTree.link(mst, graph, mst_edge_map)
        hg.CptBinaryHierarchy.link(tree, mst_edge_map, mst)

    if minima_altitudes is not None:
        altitudes = minima_altitudes[altitudes]

    return tree, altitudes
