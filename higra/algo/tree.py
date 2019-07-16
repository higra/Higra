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


@hg.argument_helper(hg.CptHierarchy)
def reconstruct_leaf_data(tree, altitudes, deleted_nodes=None, leaf_graph=None):
    """
    Each leaf of the tree takes the altitude of its closest non deleted ancestor.

    The root node is never deleted.
    In a component tree, leaves are always deleted.

    If :attr:`deleted_nodes` is ``None`` then its default value is set to `np.zeros((tree.numvertices(),)`
    (no nodes are deleted).

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param altitudes: node altitudes of the input tree
    :param deleted_nodes: binary node weights indicating which nodes are deleted (optional)
    :param leaf_graph: graph of the tree leaves (optional, deduced from :class:`~higra.CptHierarchy`)
    :return: Leaf weights
    """

    if deleted_nodes is None:
        if tree.category() == hg.TreeCategory.PartitionTree:
            leaf_weights = altitudes[0:tree.num_leaves(), ...]
        elif tree.category() == hg.TreeCategory.ComponentTree:
            parents = tree.parents()
            leaf_weights = altitudes[parents[np.arange(tree.num_leaves())], ...]
    else:
        if tree.category() == hg.TreeCategory.ComponentTree:
            deleted_nodes[:tree.num_leaves()] = True

        reconstruction = hg.propagate_sequential(tree, altitudes, deleted_nodes)
        leaf_weights = reconstruction[0:tree.num_leaves(), ...]

    if leaf_graph is not None:
        leaf_weights = hg.delinearize_vertex_weights(leaf_weights, leaf_graph)

    return leaf_weights


@hg.argument_helper(hg.CptHierarchy)
def labelisation_horizontal_cut_from_threshold(tree, altitudes, threshold, leaf_graph=None):
    """
    Labelize tree leaves according to an horizontal cut of the tree given by its altitude.

    Two leaves are in the same region (ie. have the same label) if
    the altitude of their lowest common ancestor is strictly greater
    than the specified threshold.

    :param tree: input tree (deduced from :class:`~higra.CptHierarchy`)
    :param altitudes: node altitudes of the input tree
    :param threshold: a threshold level
    :param leaf_graph: graph of the tree leaves (optional, deduced from :class:`~higra.CptHierarchy`)
    :return: Leaf labels
    """

    leaf_labels = hg.cpp._labelisation_horizontal_cut_from_threshold(tree, float(threshold), altitudes)

    if leaf_graph is not None:
        leaf_labels = hg.delinearize_vertex_weights(leaf_labels, leaf_graph)

    return leaf_labels


@hg.argument_helper(hg.CptHierarchy)
def labelisation_horizontal_cut_from_num_regions(tree, altitudes, num_regions, mode="at_least", leaf_graph=None):
    """
    Labelize tree leaves according to an horizontal cut of the tree given by its number of regions.

    If :attr:`mode` is ``"at_least"`` (default), the the smallest horizontal cut having at least the given number of
    regions is considered.
    If :attr:`mode` is ``"at_most"``, the the largest horizontal cut having at most the given number of
    regions is considered.

    :param tree: input tree (deduced from :class:`~higra.CptHierarchy`)
    :param altitudes: node altitudes of the input tree
    :param num_regions: a number of regions
    :param mode: ``"at_least"`` or ``"at_most"``
    :param leaf_graph: graph of the tree leaves (optional, deduced from :class:`~higra.CptHierarchy`)
    :return: Leaf labels
    """

    num_regions = int(num_regions)

    if mode == "at_least":
        modeb = True
    elif mode == "at_most":
        modeb = False
    else:
        raise ValueError("Incorrect mode")

    hc = hg.HorizontalCutExplorer(tree, altitudes)
    cut = hc.horizontal_cut_from_num_regions(num_regions, modeb)

    leaf_labels = cut.labelisation_leaves(tree)

    if leaf_graph is not None:
        leaf_labels = hg.delinearize_vertex_weights(leaf_labels, leaf_graph)

    return leaf_labels


@hg.argument_helper(hg.CptHierarchy)
def labelisation_hierarchy_supervertices(tree, altitudes, leaf_graph=None, handle_rag=True):
    """
    Labelize the tree leaves into supervertices.

    Two leaves are in the same supervertex if they have a common ancestor of altitude 0.

    This functions guaranties that the labels are in the range [0, num_supervertices-1].

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param altitudes: node altitudes of the input tree
    :param leaf_graph:  graph of the tree leaves (optional, deduced from :class:`~higra.CptHierarchy`)
    :param handle_rag: if True and the provided tree has been built on a region adjacency graph, then the labelisation corresponding to the rag regions is returned.
    :return: Leaf labels

    """

    if hg.CptRegionAdjacencyGraph.validate(leaf_graph) and handle_rag:
        return hg.CptRegionAdjacencyGraph.construct(leaf_graph)["vertex_map"]

    leaf_labels = hg.cpp._labelisation_hierarchy_supervertices(tree, altitudes)

    if leaf_graph is not None:
        leaf_labels = hg.delinearize_vertex_weights(leaf_labels, leaf_graph)

    return leaf_labels


@hg.argument_helper(hg.CptHierarchy)
def filter_non_relevant_node_from_tree(tree, altitudes, non_relevant_functor, leaf_graph):
    """
    Filter the given tree according to a functor telling if nodes are relevant or not.

    The given tree is first transformed into a binary tree (arbitrary choices are made).
    In a binary a tree, each inner node (non leaf node) is associated to the frontier separating its two children.
    If a the frontier associated to a node is considered as non relevant (for example because on of the two children
    of the node is too small) then the corresponding frontier is removed effectively merging its two children.

    This function returns a binary partition tree such that:

        - the frontiers associated to nodes marked *non-relevant* do not exist anymore;
        - the regions of the new tree are either regions of the initial tree or regions obtained by merging adjacent
          regions of the initial tree.

    :attr:`non_relevant_functor` must be a function that accepts two arguments, a binary tree and its node altitudes,
    and must return a boolean node attribute for the given tree (ie a 1d array of boolean-ish values of size
    ``tree.num_vertices()``. A value of ``True`` is interpreted as *this node is not relevant and its associated
    frontier must be removed*.

    :See:

        :func:`~higra.filter_small_nodes_from_tree`
        :func:`~higra.filter_weak_frontier_nodes_from_tree`

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param altitudes: node altitudes of the input tree
    :param non_relevant_functor: a function that computes an attribute on a binary tree
    :param leaf_graph: graph of the tree leaves (deduced from :class:`~higra.CptHierarchy`)
    :return: a binary tree (Concept :class:`~higra.CptBinaryHierarchy`) and its node altitudes
    """

    if not hg.CptBinaryHierarchy.validate(tree):
        saliency = hg.saliency(tree, altitudes, leaf_graph)
        tree, altitudes = hg.bpt_canonical(leaf_graph, saliency)

    mst = hg.CptBinaryHierarchy.get_mst(tree)
    deleted_frontier_nodes = non_relevant_functor(tree, altitudes)

    mst_edge_weights = altitudes[tree.num_leaves():]
    mst_edge_weights[deleted_frontier_nodes[tree.num_leaves():]] = 0
    return hg.bpt_canonical(mst, mst_edge_weights)


@hg.argument_helper(hg.CptHierarchy)
def filter_small_nodes_from_tree(tree, altitudes, size_threshold, leaf_graph):
    """
    Filter the given tree according to node size:

    This function returns a binary partition tree such that:

        - it does not contain any region whose size is below the given threshold;
        - the regions of the new tree are either regions of the initial tree or regions obtained by merging adjacent
          regions of the initial tree.

    :See:

        :func:`~higra.filter_non_relevant_node_from_tree`

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param altitudes: node altitudes of the input tree
    :param size_threshold: regions whose size is smaller than this threshold will be removed (see :func:`~higra.attribute_area`)
    :param leaf_graph: graph of the tree leaves (deduced from :class:`~higra.CptHierarchy`)
    :return: a binary tree (Concept :class:`~higra.CptBinaryHierarchy`) and its node altitudes
    """

    def non_relevant_functor(tree, _):
        area = hg.attribute_area(tree)
        return hg.accumulate_parallel(tree, area, hg.Accumulators.min) < size_threshold

    return filter_non_relevant_node_from_tree(tree, altitudes, non_relevant_functor, leaf_graph)


@hg.argument_helper(hg.CptHierarchy)
def filter_weak_frontier_nodes_from_tree(tree, altitudes, edge_weights, strength_threshold, leaf_graph):
    """
    Filter the given tree according to the frontier strength.

    The strength of a frontier is defined as the mean weights of the edges crossing the frontier
    (see :func:`~higra.attribute_frontier_strength`).

    This function returns a binary partition tree such that:

        - it does not contain any contour whose strength is lower than the given threshold;
        - the regions of the new tree are either regions of the initial tree or regions obtained by merging adjacent
          regions of the initial tree.

    :See:

        :func:`~higra.filter_non_relevant_node_from_tree`

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param altitudes: node altitudes of the input tree
    :param leaf_graph: graph of the tree leaves (deduced from :class:`~higra.CptHierarchy`)
    :param edge_weights: edge weights of the leaf graph
    :param strength_threshold: regions whose associated frontier strength is smaller than the given threshold are
        removed (see :func:`~higra.attribute_frontier_strength`)
    :return: a binary tree (Concept :class:`~higra.CptBinaryHierarchy`) and its node altitudes
    """

    def non_relevant_functor(tree, _):
        return hg.attribute_frontier_strength(tree, edge_weights, leaf_graph) < strength_threshold

    return filter_non_relevant_node_from_tree(tree, altitudes, non_relevant_functor, leaf_graph)


@hg.argument_helper(hg.CptHierarchy)
def binary_labelisation_from_markers(tree, object_marker, background_marker, leaf_graph=None):
    """
    Given two binary markers :math:`o` (object) and :math:`b` (background) (given by their indicator functions)
    on the leaves of a tree :math:`T`, the corresponding binary labelization of the leaves of :math:`T` is defined as
    the union of all the nodes intersecting :math:`o` but not :math:`b`:

    .. math::

        res = \\bigcup \{R \in T \mid R \cap o \\neq \emptyset, \\textrm{ and } R \cap b = \emptyset\}

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param object_marker: indicator function of the object marker: 1d array of size tree.num_leaves() where non zero values correspond to the object marker
    :param background_marker: indicator function of the background marker: 1d array of size tree.num_leaves() where non zero values correspond to the background marker
    :param leaf_graph: graph on the leaves of the input tree (optional, deduced from :class:`~higra.CptHierarchy`)
    :return: Leaf labels
    """

    if leaf_graph is not None:
        object_marker = hg.linearize_vertex_weights(object_marker, leaf_graph)
        background_marker = hg.linearize_vertex_weights(background_marker, leaf_graph)

    object_marker, background_marker = hg.cast_to_common_type(object_marker, background_marker)
    labels = hg.cpp._binary_labelisation_from_markers(tree, object_marker, background_marker)

    if leaf_graph is not None:
        labels = hg.delinearize_vertex_weights(labels, leaf_graph)

    return labels


@hg.argument_helper(hg.CptHierarchy)
def sort_hierarchy_with_altitudes(tree, altitudes):
    """
    Sort the nodes of a tree according to their altitudes.
    The altitudes must be increasing, i.e. for any nodes :math:`i, j` such that :math:`j` is an ancestor of :math:`i`,
    then :math:`altitudes[i] \leq altitudes[j]`.

    The result is a new tree and a node map, isomorph to the input tree such that for any nodes :math:`i` and
    :math:`j`, :math:`i<j \Rightarrow altitudes[node\_map[i]] \leq altitudes[node\_map[j]]`

    The latter condition is stronger than the original condition on the altitudes as :math:`j` is an ancestor of
    :math:`i` implies :math:`i<j` while the converse is not true.

    The returned "node_map" is an array that maps any node index :math:`i` of the new tree,
    to the index of this node in the original tree.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param altitudes: node altitudes of the input tree
    :return: the sorted tree (Concept :class:`~higra.CptHierarchy`), its node altitudes, and the node map
    """
    new_tree, node_map = hg.cpp._sort_hierarchy_with_altitudes(tree, altitudes)
    new_altitudes = altitudes[node_map]

    return new_tree, new_altitudes, node_map
