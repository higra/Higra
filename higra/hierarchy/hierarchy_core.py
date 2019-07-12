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


def bpt_canonical(graph, edge_weights):
    """
    Computes the canonical binary partition tree (binary tree by altitude ordering) of the given weighted graph.
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


def quasi_flat_zone_hierarchy(graph, edge_weights):
    """
    Computes the quasi flat zone hierarchy of the given weighted graph.
    The nodes of the quasi flat zone hierarchy corresponds to the connected components of all the possible
    thresholds of the edge weights.

    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    res = hg.cpp._quasi_flat_zone_hierarchy(graph, edge_weights)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def simplify_tree(tree, deleted_vertices, process_leaves=False):
    """
    Creates a copy of the given tree and deletes the vertices `i` of the tree such that `deletedVertices[i]` is true.

    The returned ``node_map`` is an array that maps any node index :math:`i` of the new tree,
    to the index of the corresponding node in the original tree.

    :param tree: input tree
    :param deleted_vertices: boolean valuation of the input tree nodes
    :param process_leaves: If false, a leaf vertex `v` will never be removed disregarding the value of `deletedVertices[v]`
    :return: a tree (Concept :class:`~higra.CptHierarchy` if input tree already satisfied this concept) and the node map
    """

    res = hg.cpp._simplify_tree(tree, deleted_vertices, process_leaves)
    new_tree = res.tree()
    node_map = res.node_map()

    if hg.CptHierarchy.validate(tree):
        hg.CptHierarchy().link(new_tree, hg.CptHierarchy.get_leaf_graph(tree))

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


def canonize_hierarchy(tree, altitudes):
    """
    Removes consecutive tree nodes with equal altitudes.

    The new tree is composed of the inner nodes :math:`n` of the input tree such that
    :math:`altitudes[n] \\neq altitudes[tree.parent(n)]` or :math:`n = tree.root(n)`.

    For example, applying this function to the result of :func:`~higra.bpt_canonical` on an edge weighted graph
    is the same as computing the :func:`~higra.quasi_flat_zone_hierarchy` of the same edge weighted graph.

    :param tree: input tree
    :param altitudes: altitudes of the vertices of the tree
    :return: a tree (Concept :class:`~higra.CptHierarchy` if input tree already satisfied this concept)
            and its node altitudes
    """
    ctree, node_map = hg.simplify_tree(tree, altitudes == altitudes[tree.parents()])
    return ctree, altitudes[node_map]


def tree_2_binary_tree(tree):
    """
    Transforms a tree into a binary tree.

    Each non-leaf node of the input tree must have at least 2 children!

    Whenever a non-leaf node :math:`n` with :math:`k > 2` children is found:

        - an extra node :math:`m` is created;
        - the first 2 children of :math:`n` become children of the new node :math:`m`; and
        - the new node :math:`m` becomes the first child of :math:`n`.

    The number of children of :math:`n` is thus reduced by 1.
    This operation is repeated :math:`k-2` times, i.e. until :math:`n` has only 2 children.

    The returned ``node_map`` is an array that maps any node index :math:`i` of the new tree,
    to the index of the corresponding node in the original tree.

    :Complexity:

    This algorithms runs in linear time :math:`O(tree.num\_vertices())`.

    :Examples:


    Compute the watershed hierarchy by area of an edge weighted graph and
    get the corresponding binary hierarchy. The returned ``node_map`` enables
    to recover the altitudes of the new hierarchy from the altitudes of the input
    hierarchy.

    .. code-block:: python

        tree, altitudes = watershed_hierarchy_by_area(graph, edge_weights)
        new_tree, node_map = tree_2_binary_tree(tree)
        new_altitudes = altitudes[node_map]

    :param tree: Input tree
    :return: a tree (Concept :class:`~higra.CptHierarchy` if input tree already satisfied this concept) and the node map
    """

    assert np.all(tree.num_children() >= 2), "Each non-leaf node of the input tree must have at least 2 children!"

    res = hg.cpp._tree_2_binary_tree(tree)
    ntree = res.tree()
    node_map = res.node_map()

    if hg.CptHierarchy.validate(tree):
        hg.CptHierarchy().link(ntree, hg.CptHierarchy.get_leaf_graph(tree))

    return ntree, node_map
