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


def bpt_canonical(graph, edge_weights):
    """
    Compute the canonical binary partition tree (binary tree by altitude ordering) of the given weighted graph.
    This is also known as single/min linkage clustering.

    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :return: a tree (Concept :class:`~higra.CptBinaryHierarchy`) and its node altitudes
    """
    res = hg.cpp._bpt_canonical(graph, edge_weights)
    tree = res.tree()
    altitudes = res.altitudes()
    mst = res.mst()
    mst_edge_map = res.mst_edge_map()

    if hg.CptMinimumSpanningTree.validate(graph):
        leaf_graph = hg.CptMinimumSpanningTree.construct(graph)["base_graph"]
    else:
        leaf_graph = graph

    hg.CptMinimumSpanningTree.link(mst, leaf_graph, mst_edge_map)
    hg.CptHierarchy.link(tree, leaf_graph)
    hg.CptBinaryHierarchy.link(tree, mst)

    return tree, altitudes


def quasi_flat_zones_hierarchy(graph, edge_weights):
    """
    Compute the quasi flat zones hierarchy of the given weighted graph.
    The nodes of the quasi flat zones hierarchy corresponds to the connected components of all the possible
    thresholds of the edge weights.

    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    res = hg.cpp._quasi_flat_zones_hierarchy(graph, edge_weights)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def simplify_tree(tree, deleted_vertices, process_leaves=False):
    """
    Creates a copy of the given tree and deletes the vertices `i` of the tree such that `deletedVertices[i]` is true.

    The returned "node_map" of the returned tree is an array that maps any node index i of the new tree,
    to the index of this node in the original tree.

    :param tree: input tree
    :param deleted_vertices: boolean valuation of the input tree nodes
    :param process_leaves: If false, a leaf vertex `v` will never be removed disregarding the value of `deletedVertices[v]`
    :return: a simplified tree (Concept :class:`~higra.CptHierarchy`) and the node map
    """

    res = hg.cpp._simplify_tree(tree, deleted_vertices, process_leaves)
    new_tree = res.tree()
    node_map = res.node_map()

    if hg.CptHierarchy.validate(tree):
        hg.CptHierarchy().link(tree, hg.CptHierarchy.construct(tree)["leaf_graph"])

    return new_tree, node_map


@hg.argument_helper(hg.CptHierarchy)
def saliency(tree, altitudes, leaf_graph, handle_rag=True):
    """
    The saliency map of the input hierarchy :math:`(tree, altitudes)` for the leaf graph :math:`g` is an array of
    edge weights :math:`sm` for :math:`g` such that for each pair of adjacent vertices :math:`(i,j)` in :math:`g`,
    :math:`sm(i,j)` is equal to the ultra-metric distance between :math:`i` and :math:`j` corresponding to the hierarchy.

    Formally, this is computed using the following property: :math:`sm(i,j) = altitudes(lowest\_common\_ancestor_{tree}(i,j))`.

    Complexity: :math:`\mathcal{O}(n\log(n) + m)` with :math:`n` the number of vertices in the tree and :math:`m` the number of edges in the graph.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param altitudes: altitudes of the vertices of the tree
    :param leaf_graph: graph whose vertex set is equal to the leaves of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :param handle_rag: if tree has been constructed on a rag, then saliency values will be propagated to the original graph, hence leading to a saliency on the original graph and not on the rag
    :return: 1d array of edge weights
    """
    lca_map = hg.attribute_lca_map(tree, leaf_graph=leaf_graph)

    sm = altitudes[lca_map]
    if hg.CptRegionAdjacencyGraph.validate(leaf_graph) and handle_rag:
        sm = hg.rag_back_project_edge_weights(leaf_graph, sm)

    return sm
