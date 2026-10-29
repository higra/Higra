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
from ..hg_utils import _as_index_array, _validate_edge_weights


def bpt_canonical(graph, edge_weights=None, sorted_edge_indices=None, return_altitudes=True, compute_mst=True,
                  *, method="kruskal", **kwargs):
    """
    Computes the *canonical binary partition tree*, also called *binary partition tree by altitude ordering* or
    *connectivity constrained single min/linkage clustering* of the given graph.

    :Definition:

    The following definition is adapted from:

        Cousty, Jean, Laurent Najman, Yukiko Kenmochi, and Silvio Guimarães.
        `"Hierarchical segmentations with graphs: quasi-flat zones, minimum spanning trees, and saliency maps."
        <https://hal.archives-ouvertes.fr/hal-01344727/document>`_
        Journal of Mathematical Imaging and Vision 60, no. 4 (2018): 479-502.

    Let :math:`G=(V,E)` be an undirected graph, let :math:`\\prec` be a total order on :math:`E`, and let :math:`e_k` be
    the edge in :math:`E` that has exactly :math:`k` smaller edges according to :math:`\\prec`: we then say that :math:`k`
    is the rank of :math:`e_k` (for :math:`\\prec`).
    The *canonical binary partition hierarchy* of :math:`G` for :math:`\\prec` is defined as the sequence of nested partitions:

    - :math:`P_0 = \{\{v\}, v\in V\}`, the finest partion is composed of every singleton of :math:`V`; and
    - :math:`P_n = (P_{n-1} \\backslash \{P_{n-1}^x, P_{n-1}^y\}) \cup (P_{n-1}^x \cup P_{n-1}^y)` where :math:`e_n=\{x,y\}`
      and :math:`P_{n-1}^x` and :math:`P_{n-1}^y` are the regions of  :math:`P_{n-1}` that contain :math:`x` and :math:`y`
      respectively.

    At the step :math:`n`, we remove the regions at the two extremities of the :math:`n`-th smallest edge
    and we add their union. Note that we may have :math:`P_n = P_{n-1}` if both extremities of the edge :math:`e_n`
    were in a same region of :math:`P_{n-1}`. Otherwise, :math:`P_n` is obtained by merging two regions of :math:`P_{n-1}`.

    The *canonical binary partition tree* is then the tree representing the merging steps in this sequence,
    it is thus binary. Each merging step, and thus each non leaf node of the tree, is furthermore associated to a
    specific edge of the graph, called *a building edge* that led to this merge. It can be shown that the set of all
    building edges associated to a canonical binary partition tree is a minimum spanning tree of the graph for the given
    edge ordering :math:`\\prec`.

    The map that associates every non leaf node of the canonical binary partition tree to its building edge is called
    the *mst_edge_map*. In practice this map is represented by an array of size :math:`tree.num\_vertices() - tree.num\_leaves()`
    and, for any internal node :math:`i` of the tree, :math:`mst\_edge\_map[i - tree.num\_leaves()]` is equal to the index
    of the building edge in :math:`G` associated to :math:`i`.

    The ordering :math:`\\prec` can be specified explicitly by providing the array of indices :attr:`sorted_edge_indices`
    that sort the edges, or implicitly by providing the array of edge weights :attr:`edge_weights`. In this case,
    the edge order is equivalent to ``hg.arg_sort(edge_weights, stable=True)``. If :attr:`edge_weights`
    is an array with more than 1 dimension, a lexicographic ordering is used.

    If requested, altitudes associated to the nodes of the canonical binary partition tree are computed as follows:

      - if :attr:`edge_weights` are provided, the altitude of a non-leaf node is equal to the edge weight of its
        building edge; and
      - otherwise, the altitude of a non-leaf node is equal to the rank of its building edge.

    The altitude of a leaf node is always equal to 0.

    :Example:

    .. figure:: /fig/canonical_binary_partition_tree_example.svg
        :alt: Example of a binary partition tree by altitude ordering
        :align: center

        Given an edge weighted graph :math:`G`, the binary partition tree by altitude ordering :math:`T` (in blue) is
        associated to a minimum spanning tree :math:`S` of :math:`G` (whose edges are thick and gray). Each leaf node of
        the tree corresponds to a vertex of :math:`G` while each non-leaf node :math:`n_i` of :math:`T` corresponds to a
        building edge of :math:`T` which belongs to the minimum spanning tree :math:`S`. The association between the non-leaf
        nodes and the minimum spanning tree edges, called *mst_edge_map*, is depicted by green arrows .


    The above figure corresponds to the following code (note that vertex indices start at 0 in the code):

        >>> g = hg.UndirectedGraph(5)
        >>> g.add_edges((0, 0, 1, 1, 1, 2, 3),
        >>>             (1, 2, 2, 3, 4, 4, 4))
        >>> edge_weights = np.asarray((4, 6, 3, 7, 11, 8, 5))
        >>> tree, altitudes = hg.bpt_canonical(g, edge_weights)
        >>> tree.parents()
        array([6, 5, 5, 7, 7, 6, 8, 8, 8])
        >>> altitudes
        array([0, 0, 0, 0, 0, 3, 4, 5, 7])
        >>> tree.mst_edge_map
        array([2, 0, 6, 3])
        >>> tree.mst.edge_list()
        (array([1, 0, 3, 1]), array([2, 1, 4, 3]))

    An object of type UnidrectedGraph is not necessary:

        >>> edge_weights = np.asarray((4, 6, 3, 7, 11, 8, 5))
        >>> sources = (0, 0, 1, 1, 1, 2, 3)
        >>> targets = (1, 2, 2, 3, 4, 4, 4)
        >>> tree, altitudes = hg.bpt_canonical((sources, targets, 5), edge_weights)
        >>> tree.parents()
        array([6, 5, 5, 7, 7, 6, 8, 8, 8])
        >>> altitudes
        array([0, 0, 0, 0, 0, 3, 4, 5, 7])
        >>> tree.mst_edge_map
        array([2, 0, 6, 3])


    
    Two construction algorithms are available: Kruskal's algorithm and Waterfall–Boruvka. See the references below for details on the algorithms and specific parameters.

    - Kruskal's algorithm is the default, it is based on the classical Kruskal's minimum spanning tree algorithm. It is based on the classical Kruskal's minimum spanning tree, see :func:`bpt_canonical_kruskal` for details.
    - Waterfall–Boruvka is based on the Boruvka minimum spanning tree algorithm algorithm, see :func:`bpt_canonical_boruvka` for details.
    
    Waterfall–Boruvka is better suited for massively parallel execution, and can be faster than Kruskal's algorithm for large graphs. 


    :param graph: input graph or triplet of two arrays and an integer (sources, targets, num_vertices)
           defining all the edges of the graph and its number of vertices.
    :param edge_weights: edge weights of the input graph (may be omitted if :attr:`sorted_edge_indices` is given).
    :param sorted_edge_indices: array of indices that sort the edges of the input graph by increasing weight (may be
           omitted if :attr:`edge_weights` is given).
    :param return_altitudes: if ``True`` an array representing the altitudes of the tree vertices is returned.
           (default: ``True``).
    :param compute_mst: if ``True`` and if the input is a graph object computes an explicit undirected graph
           representing the minimum spanning tree associated to the hierarchy, accessible through the
           :class:`~higra.CptBinaryHierarchy` Concept (e.g. with ``tree.mst``). (default: ``True``).
    :param method: ``"kruskal"`` or ``"boruvka"``
    :param kwargs: keyword arguments for the selected algorithm
    :return: a tree (Concept :class:`~higra.CptBinaryHierarchy` if the input is a graph object),
             and, if :attr:`return_altitudes` is ``True``, its node altitudes
    """

    if not isinstance(method, str) or method not in ("kruskal", "boruvka"):
        raise ValueError('method must be "kruskal" or "boruvka".')
    if edge_weights is None and sorted_edge_indices is None:
        raise ValueError("edge_weights and sorted_edge_indices cannot be both equal to None.")

    input_is_graph_object = False

    if hg.has_method(graph, "edge_list") and hg.has_method(graph, "num_vertices"):
        input_is_graph_object = True
        sources, targets = graph.edge_list()
        num_vertices = graph.num_vertices()
    else:
        try:
            sources, targets, num_vertices = graph
        except Exception as e:
            raise ValueError("Invalid graph input.") from e

    sources = _as_index_array(sources, "sources")
    targets = _as_index_array(targets, "targets")
    if not isinstance(num_vertices, (int, np.integer)) or isinstance(num_vertices, (bool, np.bool_)):
        raise ValueError("num_vertices must be an integer.")

    weights = None
    if edge_weights is not None:
        edge_weights = _validate_edge_weights(edge_weights, sources.size)
        features = int(np.prod(edge_weights.shape[1:], dtype=np.intp))
        weights = np.asarray(edge_weights, dtype=edge_weights.dtype.newbyteorder("="))
        weights = weights.reshape((sources.size, features))

    order = None
    if sorted_edge_indices is not None:
        order = _as_index_array(sorted_edge_indices, "sorted_edge_indices")
        sorted_edge_indices = np.asarray(sorted_edge_indices)
        if sorted_edge_indices.dtype.kind not in "iu":
            sorted_edge_indices = order

    algorithm = _bpt_canonical_kruskal if method == "kruskal" else _bpt_canonical_boruvka
    parents, building_edge_map = algorithm(sources, targets, num_vertices, weights, order, **kwargs)
    tree = hg.Tree(parents)

    if return_altitudes:
        if edge_weights is None:
            edge_weights = np.empty_like(sorted_edge_indices)
            edge_weights[sorted_edge_indices] = np.arange(sorted_edge_indices.size)

        if edge_weights.ndim == 1:
            altitudes = np.zeros((tree.num_vertices(),), dtype=edge_weights.dtype)
            altitudes[num_vertices:] = edge_weights[building_edge_map]
        else:
            shape = [tree.num_vertices()] + list(edge_weights.shape[1:])
            altitudes = np.zeros(shape, dtype=edge_weights.dtype)
            altitudes[num_vertices:, ...] = edge_weights[building_edge_map, ...]

    mst_edge_map = building_edge_map
    if input_is_graph_object:
        # Input-MST metadata indexes its immediate base graph.
        if hg.CptMinimumSpanningTree.validate(graph):
            input_mst = hg.CptMinimumSpanningTree.construct(graph)
            leaf_graph = input_mst["base_graph"]
            mst_edge_map = input_mst["mst_edge_map"][building_edge_map]
        else:
            leaf_graph = graph

        hg.CptHierarchy.link(tree, leaf_graph)

    if compute_mst and input_is_graph_object:
        mst = hg.subgraph(graph, building_edge_map)
        hg.CptMinimumSpanningTree.link(mst, leaf_graph, mst_edge_map)
    else:
        mst = None

    hg.CptBinaryHierarchy.link(tree, mst_edge_map, mst)

    if not return_altitudes:
        return tree
    else:
        return tree, altitudes


def bpt_canonical_kruskal(graph, edge_weights=None, sorted_edge_indices=None,
                          return_altitudes=True, compute_mst=True):
    """
    Construct a canonical binary partition tree using Kruskal's algorithm.

    See :func:`bpt_canonical` for the hierarchy definition, input forms, edge
    ordering, altitudes and MST metadata. Equal weights are ordered by increasing
    edge ID unless ``sorted_edge_indices`` supplies an explicit order.

    :Complexity:

    The algorithm used is based on Kruskal's minimum spanning tree algorithm and is described in:

        Laurent Najman, Jean Cousty, Benjamin Perret.
        `Playing with Kruskal: Algorithms for Morphological Trees in Edge-Weighted Graphs
        <https://hal.archives-ouvertes.fr/file/index/docid/798621/filename/ismm2013-algo.pdf>`_.
        ISMM 2013: 135-146.

    If :attr:`sorted_edge_indices` is provided the algorithm runs in quasi linear :math:`\mathcal{O}(n \\alpha(n))`,
    with :math:`n` the number of elements in the graph and with :math:`\\alpha` the inverse of the Ackermann function.
    Otherwise, the computation time is dominated by the sorting of the edge weights which is performed in linearithmic
    :math:`\mathcal{O}(n \log(n))` time.

    :param graph: graph object or ``(sources, targets, num_vertices)`` tuple
    :param edge_weights: edge weights, with optional trailing feature dimensions
    :param sorted_edge_indices: edge IDs in construction order (optional)
    :param return_altitudes: return node altitudes alongside the tree
    :param compute_mst: construct the explicit MST for graph objects
    :return: binary partition tree, and node altitudes if requested
    """
    return bpt_canonical(graph, edge_weights, sorted_edge_indices, return_altitudes, compute_mst,
                         method="kruskal")


def bpt_canonical_boruvka(graph, edge_weights=None, sorted_edge_indices=None,
                          return_altitudes=True, compute_mst=True, stable=True,
                          *, compact_edges=False):
    """
    Construct a canonical binary partition tree using Waterfall–Boruvka.

    See :func:`bpt_canonical` for the hierarchy definition, input forms, edge
    ordering, altitudes and MST metadata. Equal weights are ordered by increasing
    edge ID unless ``sorted_edge_indices`` supplies an explicit order.

    The algorithm is based on Boruvka's minimum spanning tree algorithm and is described in:

        Quentin Lebon, Jean Cousty, Thierry Grandpierre, Benjamin Perret
        `Waterfall-Borvka-Based Algorithm for Binary Partition Tree. 
        <https://hal.science/hal-05059071v1/file/Waterfall_Boruvka_based_algorithm_for_binary_partiton_tree.pdf>`_.
        DGMM 2025: 328-340.


    With ``stable=True``, parents and the ordered MST edge map match
    :func:`bpt_canonical_kruskal` for the same edge order: 
    the returned node altitudes are sorted in ascending order. 
    ``stable=False``  selects basin numbering: does not guarantee that the returned node altitudes are sorted 
    (the nodes are still in topological order), but may be faster.
    Note that the final tree is always isomorphic to the tree returned by :func:`bpt_canonical_kruskal`.

    Compact active edges each round instead of scanning all edges (default: ``False``) does not change the output, but may be faster.

    :Complexity:

    The serial work bound is :math:`O((n+m) \\log n)` for scalar weights, with :math:`O(n+m)`
    auxiliary storage, where ``n`` and ``m`` are vertex and edge counts.

    :param graph: graph object or ``(sources, targets, num_vertices)`` tuple
    :param edge_weights: real integer or float32/float64 edge array (optional)
    :param sorted_edge_indices: permutation of all edge IDs (optional)
    :param return_altitudes: return node altitudes alongside the tree
    :param compute_mst: construct the explicit MST for graph objects
    :param stable: exact numbering by default; ``False`` selects basin numbering
    :param compact_edges: compact active edges each round instead of scanning all
        edges (default: ``False``); does not change the output
    :return: binary partition tree, and node altitudes if requested
    :raises ValueError: invalid shapes, dtypes, NaNs, or integer indices
    :raises RuntimeError: invalid permutation, endpoints, empty or disconnected graph
    """
    return bpt_canonical(graph, edge_weights, sorted_edge_indices, return_altitudes, compute_mst,
                         method="boruvka", stable=stable, compact_edges=compact_edges)


def _bpt_canonical_kruskal(sources, targets, num_vertices, edge_weights, sorted_edge_indices):
    if sorted_edge_indices is None:
        sort_weights = edge_weights[:, 0] if edge_weights.shape[1] == 1 else edge_weights
        sorted_edge_indices = hg.arg_sort(sort_weights, stable=True)
    return hg.cpp._bpt_canonical(sources, targets, sorted_edge_indices, num_vertices)


def _bpt_canonical_boruvka(sources, targets, num_vertices, edge_weights, sorted_edge_indices,
                           stable=True, *, compact_edges=False):
    if sorted_edge_indices is not None:
        return hg.cpp._bpt_canonical_boruvka_from_order(
            sources, targets, num_vertices, sorted_edge_indices, stable, compact_edges)
    return hg.cpp._bpt_canonical_boruvka(
        sources, targets, num_vertices, edge_weights, stable, compact_edges)


def quasi_flat_zone_hierarchy(graph, edge_weights):
    """
    Computes the quasi flat zone hierarchy of the given weighted graph.
    The nodes of the quasi flat zone hierarchy corresponds to the connected components of all the possible
    thresholds of the edge weights.

    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    tree, altitudes = hg.cpp._quasi_flat_zone_hierarchy(graph, edge_weights)

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def simplify_tree(tree, deleted_vertices, process_leaves=False):
    """
    Creates a copy of the given tree and deletes the vertices :math:`i` of the tree such that :math:`deletedVertices[i]`
    is ``True``.

    The returned ``node_map`` is an array that maps any node index :math:`i` of the new tree,
    to the index of the corresponding node in the original tree.


    :param tree: input tree
    :param deleted_vertices: boolean valuation of the input tree nodes
    :param process_leaves: If ``False``, a leaf vertex :math:`v` will never be removed disregarding the value of
                            :math:`deletedVertices[v]`. If ``True``, leaves node may be removed. Note that in this
                            case, a reordering of the nodes may be necessary, which is a more complex and slower operation.
    :return: a tree (Concept :class:`~higra.CptHierarchy` if input tree already satisfied this concept) and the node map
    """

    if len(deleted_vertices.shape) != 1 or deleted_vertices.shape[0] != tree.num_vertices():
        raise ValueError("Parameter 'deleted_vertices' must an array of shape [tree.num_vertices()].")

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


def canonize_hierarchy(tree, altitudes, return_node_map=False):
    """
    Removes consecutive tree nodes with equal altitudes.

    The new tree is composed of the inner nodes :math:`n` of the input tree such that
    :math:`altitudes[n] \\neq altitudes[tree.parent(n)]` or :math:`n = tree.root(n)`.

    For example, applying this function to the result of :func:`~higra.bpt_canonical` on an edge weighted graph
    is the same as computing the :func:`~higra.quasi_flat_zone_hierarchy` of the same edge weighted graph.

    If :attr:`return_node_map` is ``True``, an extra array that maps any vertex index :math:`i` of the new tree,
    to the index of the corresponding vertex in the original tree is returned.

    :param tree: input tree
    :param altitudes: altitudes of the vertices of the tree
    :param return_node_map: if ``True``, also return the node map.
    :return: a tree (Concept :class:`~higra.CptHierarchy` if input tree already satisfied this concept)
             its node altitudes, and, if requested, its node map.
    """
    tree, node_map = hg.simplify_tree(tree, altitudes == altitudes[tree.parents()])
    new_altitudes = altitudes[node_map]
    if return_node_map:
        return tree, new_altitudes, node_map
    else:
        return tree, new_altitudes


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
