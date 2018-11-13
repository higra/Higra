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


@hg.argument_helper(hg.CptValuedHierarchy)
def reconstruct_leaf_data(altitudes, deleted_nodes, tree):
    """
    Each leaf of the tree takes the altitude of its closest non deleted ancestor.

    :param tree:
    :param deleted_nodes:
    :param altitudes:
    :return:
    """
    reconstruction = hg.propagate_sequential(altitudes,
                                             deleted_nodes,
                                             tree)
    leaf_weights = reconstruction[0:tree.num_leaves(), ...]

    if hg.CptHierarchy.validate(tree):
        leaf_graph = hg.CptHierarchy.construct(tree)["leaf_graph"]
        leaf_weights = hg.delinearize_vertex_weights(leaf_weights, leaf_graph)
        hg.CptVertexWeightedGraph.link(leaf_weights, leaf_graph)

    return leaf_weights


@hg.argument_helper(hg.CptValuedHierarchy)
def labelisation_horizontal_cut_from_threshold(altitudes, threshold, tree):
    """
    Labelize tree leaves according to an horizontal cut in the tree.

    Two leaves are in the same region (ie. have the same label) if
    the altitude of their lowest common ancestor is strictly greater
    than the specified threshold.

    :param tree:
    :param threshold:
    :param altitudes:
    :return:
    """

    leaf_labels = hg._labelisation_horizontal_cut_from_threshold(tree, float(threshold), altitudes)

    if hg.CptHierarchy.validate(tree):
        leaf_graph = hg.CptHierarchy.construct(tree)["leaf_graph"]
        leaf_labels = hg.delinearize_vertex_weights(leaf_labels, leaf_graph)
        hg.CptVertexLabeledGraph.link(leaf_labels, leaf_graph)

    return leaf_labels


@hg.argument_helper(hg.CptValuedHierarchy)
def labelisation_hierarchy_supervertices(altitudes, tree, leaf_graph=None, handle_rag=True):
    """
    Labelize the tree leaves into supervertices.

    Two leaves are in the same supervertex if they have a common ancestor of altitude 0.

    If handle_rag is True and the provided has been built on a region adjacency graph, then the labelisation
    corresponding to the rag regions is returned.

    This functions guaranties that the labels are in the range [0, num_supervertices-1].

    :param tree:
    :param altitudes:
    :param handle_rag:
    :return:
    """

    if hg.CptRegionAdjacencyGraph.validate(leaf_graph) and handle_rag:
        return hg.CptRegionAdjacencyGraph.construct(leaf_graph)["vertex_map"]

    leaf_labels = hg._labelisation_hierarchy_supervertices(tree, altitudes)

    if leaf_graph is not None:
        leaf_labels = hg.delinearize_vertex_weights(leaf_labels, leaf_graph)
        hg.CptVertexLabeledGraph.link(leaf_labels, leaf_graph)

    return leaf_labels


@hg.argument_helper(hg.CptValuedHierarchy, ("tree", hg.CptBinaryHierarchy))
def filter_binary_partition_tree(altitudes, deleted_frontier_nodes, tree, mst):
    """
    Filter the given binary partition tree according to the given list of frontiers to remove.

    In a binary a tree, each inner node (non leaf node) is associated to the frontier separating its two children.
    If this node frontier is marked for deletion then the corresponding frontier is removed
    effectively merging its two children.

    :param tree: input binary partition tree (should be constructed with the function `bpt_canonical`)
    :param deleted_frontier_nodes: a boolean array indicating for each node of the tree is its children must be merged (True) or not (False)
    :return: a filtered binary partition tree
    """

    mst_edge_weights = altitudes[tree.num_leaves():]
    mst_edge_weights[deleted_frontier_nodes[tree.num_leaves():]] = 0
    return hg.bpt_canonical(mst_edge_weights, mst)


@hg.argument_helper(hg.CptHierarchy)
def binary_labelisation_from_markers(tree, object_marker, background_marker, leaf_graph=None):
    """
    Given two binary markers o (object) and b (background) (given by their indicator functions)
    on the leaves of a tree t, the corresponding binary labelization of the leaves of t is defined as
    the union of all the nodes intersecting o but not b.

    final_object = union {R in T | R cap o neq emptyset and R cap b = emptyset}

    :param tree: input tree
    :param object_marker: indicator function of the object marker: 1d array of size tree.num_leaves() where non zero values correspond to the object marker
    :param background_marker: indicator function of the background marker: 1d array of size tree.num_leaves() where non zero values correspond to the background marker
    :return:
    """

    if leaf_graph is not None:
        object_marker = hg.linearize_vertex_weights(object_marker, leaf_graph)
        background_marker = hg.linearize_vertex_weights(background_marker, leaf_graph)

    labels = hg._binary_labelisation_from_markers(tree, object_marker, background_marker)

    if leaf_graph is not None:
        labels = hg.delinearize_vertex_weights(labels, leaf_graph)
        hg.CptVertexLabeledGraph.link(labels, leaf_graph)

    return labels
