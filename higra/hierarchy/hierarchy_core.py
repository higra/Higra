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


@hg.argument_helper(hg.CptEdgeWeightedGraph)
def bpt_canonical(edge_weights, graph):
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
    mst_edge_map = res.mst_edge_map()

    if hg.CptMinimumSpanningTree.validate(graph):
        leaf_graph = hg.CptMinimumSpanningTree.construct(graph)["base_graph"]
    else:
        leaf_graph = graph

    hg.CptMinimumSpanningTree.link(mst, leaf_graph, mst_edge_map)
    hg.CptHierarchy.link(tree, leaf_graph)
    hg.CptBinaryHierarchy.link(tree, mst)
    hg.CptValuedHierarchy.link(altitudes, tree)

    return tree, altitudes


@hg.argument_helper(hg.CptEdgeWeightedGraph)
def quasi_flat_zones_hierarchy(edge_weights, graph):
    """
    Compute the quasi flat zones hierarchy of the given weighted graph.

    :param graph:
    :param edge_weights:
    :return: Tree (with attributes "leaf_graph" and "altitudes")
    """

    res = hg._quasi_flat_zones_hierarchy(graph, edge_weights)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)
    hg.CptValuedHierarchy.link(altitudes, tree)

    return tree, altitudes


@hg.argument_helper(hg.CptValuedHierarchy)
def simplify_tree(deleted_vertices, tree):
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

    if hg.CptHierarchy.validate(tree):
        hg.CptHierarchy().link(tree, hg.CptHierarchy.construct(tree)["leaf_graph"])

    return new_tree, node_map


@hg.argument_helper(hg.CptValuedHierarchy, ("tree", "lca_map"))
def saliency(altitudes, leaf_graph, lca_map, handle_rag=True):
    """
    Compute the saliency map (ultra-metric distance) of the given tree.

    :param altitudes: altitudes of the vertices of the tree
    :param lca_map: array containing the lowest common ancestor of the source and target vertices of each edge where saliency need to be computed
    :param handle_rag: if tree has been constructed on a rag, then saliency values will be propagated to the original graph, hence leading to a saliency on the original graph and not on the rag
    :return: edge saliency corresponding to the given tree
    """
    sm = altitudes[lca_map]
    if hg.CptRegionAdjacencyGraph.validate(leaf_graph) and handle_rag:
        sm = hg.rag_back_project_edge_weights(sm, leaf_graph)
        hg.CptSaliencyMap.link(sm, hg.CptRegionAdjacencyGraph.construct(leaf_graph)["pre_graph"])
    else:
        hg.CptSaliencyMap.link(sm, leaf_graph)
    return sm
