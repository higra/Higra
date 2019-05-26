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


@hg.extend_class(hg.HorizontalCutNodes, method_name="reconstruct_leaf_data")
@hg.argument_helper(("tree", hg.CptHierarchy))
def __reconstruct_leaf_data(self, tree, altitudes, leaf_graph, handle_rag=True):
    """
    Reconstruct the cut at leaf level of the provided tree.
    A leaf of the tree is valued by the altitude of the single cut node which is an ancestor of the leaf.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param altitudes: node altitudes of the input tree
    :param leaf_graph: graph on the tree leaves (deduced from :class:`~higra.CptHierarchy`)
    :param handle_rag: if `True` and if `leaf_graph` is a region adjacency graph then the cut is given for the original graph (the pre-graph of the region adjacency graph).
    :return: leaf weights
    """
    leaf_weights = self._reconstruct_leaf_data(tree, altitudes)

    if hg.CptRegionAdjacencyGraph.validate(leaf_graph) and handle_rag:
        leaf_weights = hg.rag_back_project_vertex_weights(leaf_graph, leaf_weights)
    else:
        leaf_weights = hg.delinearize_vertex_weights(leaf_weights, leaf_graph)

    return leaf_weights


@hg.extend_class(hg.HorizontalCutNodes, method_name="graph_cut")
@hg.argument_helper(("tree", hg.CptHierarchy))
def __graph_cut(self, tree, leaf_graph, handle_rag=True):
    """
    Graph cut corresponding to the tree cut.
    The edge (i, j) has a non zero value if the closest ancestors of i and j in the cut are different.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param leaf_graph: graph on the tree leaves (deduced from :class:`~higra.CptHierarchy`)
    :param handle_rag: if `True` and if `leaf_graph` is a region adjacency graph then the cut is given for the original graph (the pre-graph of the region adjacency graph).
    :return: a graph cut
    """
    sm = self._graph_cut(tree, leaf_graph)

    if hg.CptRegionAdjacencyGraph.validate(leaf_graph) and handle_rag:
        sm = hg.rag_back_project_edge_weights(leaf_graph, sm)

    return sm


@hg.extend_class(hg.HorizontalCutNodes, method_name="labelisation_leaves")
@hg.argument_helper(("tree", hg.CptHierarchy))
def __labelisation_leaves(self, tree, leaf_graph, handle_rag=True):
    """
    Labelize tree leaves according to the horizontal cut.
    Two leaves are in the same region (ie. have the same label) if their lowest common ancestor is a subset or equal to
    one the node of the cut.,

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param leaf_graph: graph on the tree leaves (deduced from :class:`~higra.CptHierarchy`)
    :param handle_rag: if `True` and if `leaf_graph` is a region adjacency graph then the labels are given for the original graph (the pre-graph of the region adjacency graph).
    :return: a 1d array
    """
    labels = self._labelisation_leaves(tree)

    if hg.CptRegionAdjacencyGraph.validate(leaf_graph) and handle_rag:
        labels = hg.rag_back_project_vertex_weights(leaf_graph, labels)
    else:
        labels = hg.delinearize_vertex_weights(labels, leaf_graph)

    return labels


@hg.extend_class(hg.HorizontalCutExplorer, method_name="__new__")
def __make_HorizontalCutExplorer(cls, tree, altitudes):
    return cls._make_HorizontalCutExplorer(tree, altitudes)


@hg.extend_class(hg.HorizontalCutExplorer, method_name="__init__")
def __dummy_init_HorizontalCutExplorer(*_):
    pass
