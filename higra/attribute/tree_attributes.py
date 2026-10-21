############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################


import numpy as np
import higra as hg
from ..data_cache import _legacy_cache_keywords


@_legacy_cache_keywords
@hg.argument_helper(hg.CptHierarchy)
def attribute_area(tree, vertex_area=None, leaf_graph=None):
    """
    Compute the area of each node of the given tree.

    Each leaf of the tree corresponds to a vertex of the leaf graph (for example a pixel) and has an area :math:`a(v)`.
    The area :math:`A(n)` of a node :math:`n` is the total area of the region it represents, i.e. the sum of the
    areas of the leaves of the subtree rooted in :math:`n`:

    .. math::

        A(n) = \sum_{v \in leaves(n)} a(v)

    Equivalently, the area of a leaf is its own area, and the area of an internal node is the sum of the areas of
    its children:

    .. math::

        A(n) = \\begin{cases}
            a(n) & \\textrm{if } n \\textrm{ is a leaf}\\\\
            \sum_{c \in children(n)} A(c) & \\textrm{otherwise}
        \\end{cases}

    The area of the root is thus the total area of all the leaves.

    The leaf areas are determined as follows:

    - if :attr:`vertex_area` is provided, it gives the area of each leaf: a 1d array with one element per leaf or,
      if the leaf graph is a grid graph (e.g. an image), an array whose first dimensions follow the shape of the grid;
    - otherwise, if the tree has a leaf graph, the area of each leaf is given by :func:`~higra.attribute_vertex_area`
      (1 for each vertex, except for a region adjacency graph where it is the number of pixels in the region);
    - otherwise, the area of each leaf is 1, and the area of a node is then its number of leaves.

    :Complexity:

    The tree accumulation takes :math:`\mathcal{O}(N)` time, where :math:`N` is the number of nodes in the tree.
    If ``vertex_area`` is omitted and a leaf graph is available, add the cost of
    :func:`~higra.attribute_vertex_area`. For a region adjacency graph, area computation also
    accumulates over the vertices of its underlying graph(s).

    :Example:
    =======

    Consider the following binary tree with 5 leaves:

    .. code-block:: text

        3         +-----8-----+
                  |           |
        2     +---7---+       |
              |       |       |
        1   +-5-+   +-6-+     |
            |   |   |   |     |
        0   0   1   2   3     4

    .. code-block:: python

        tree = hg.Tree((5, 5, 6, 6, 8, 7, 7, 8, 8))
        hg.attribute_area(tree)
        # array([1., 1., 1., 1., 1., 2., 2., 4., 5.])

    With custom leaf areas:

    .. code-block:: python

        hg.attribute_area(tree, vertex_area=np.asarray((1, 2, 3, 4, 5)))
        # array([ 1,  2,  3,  4,  5,  3,  7, 10, 15])

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param vertex_area: area of the vertices of the leaf graph of the tree (provided by :func:`~higra.attribute_vertex_area` on `leaf_graph` )
    :param leaf_graph: (deduced from :class:`~higra.CptHierarchy`)
    :return: a 1d array representing the area of each node of the tree
    """
    if vertex_area is None:
        if leaf_graph is not None:
            vertex_area = hg.attribute_vertex_area(leaf_graph)
        else:
            vertex_area = np.ones((tree.num_leaves(),), dtype=np.float64)

    if leaf_graph is not None:
        vertex_area = hg.linearize_vertex_weights(vertex_area, leaf_graph)
    return hg.accumulate_sequential(tree, vertex_area, hg.Accumulators.sum)


@_legacy_cache_keywords
def attribute_volume(tree, altitudes, area=None, leaf_volume=None):
    """
    Compute the volume under the curve of each node the given tree.
    The volume :math:`V(n)` of a node :math:`n` is defined recursively as:

    .. math::

        V(n) = area(n) * | altitude(n) - altitude(parent(n)) | +  \sum_{c \in children(n)} V(c)

    The recursive formula applies to the non-leaf nodes. The volume of a leaf is not computed with this formula:
    it is simply equal to the corresponding value of ``leaf_volume``. If ``leaf_volume`` is ``None``,
    leaf volumes are set to 0.

    The root of the tree is its own parent: its altitude difference with its parent is null, hence the root does not
    add any volume of its own and its volume is the sum of the volumes of its children.

    If the area of the nodes is not provided, it is computed with :func:`~higra.attribute_area` on the input tree.

    :Complexity:

    The volume accumulation takes :math:`\mathcal{O}(N)` time, where :math:`N` is the number of nodes in the tree.
    If ``area`` is omitted, add the cost of :func:`~higra.attribute_area`, including any
    leaf-area computation for the underlying graph(s) of a region adjacency graph.

    :Example:
    =======
    
    Consider the following binary tree with 5 leaves:

    .. code-block:: text

        3         +-----8-----+
                  |           |
        2     +---7---+       |
              |       |       |
        1   +-5-+   +-6-+     |
            |   |   |   |     |
        0   0   1   2   3     4

    .. code-block:: python

        tree = hg.Tree((5, 5, 6, 6, 8, 7, 7, 8, 8))
        altitudes = np.asarray((0, 0, 0, 0, 0, 1, 2, 4, 5))
        hg.attribute_volume(tree, altitudes)
        # array([ 0.,  0.,  0.,  0.,  0.,  6.,  4., 14., 14.])

    With a non-zero initial volume for the leaves:

    .. code-block:: python

        hg.attribute_volume(tree, altitudes, leaf_volume=np.asarray((1, 1, 1, 1, 1)))
        # array([ 1.,  1.,  1.,  1.,  1.,  8.,  6., 18., 19.])

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param altitudes: node altitudes of the input tree (Numpy array of shape (tree.num_vertices(),))
    :param area: area of the nodes of the input hierarchy (provided by :func:`~higra.attribute_area` on `tree`)
    :param leaf_volume: initial volume of the leaves of the tree (default is 0 for all leaves) (Numpy array of shape (tree.num_leaves(),));
        useful when leaves represent pre-existing regions, for example in region adjacency graph based hierarchies
    :return: a 1d array representing the volume of each node of the tree
    """
    if area is None:
        area = hg.attribute_area(tree)

    height = np.abs(altitudes[tree.parents()] - altitudes)
    height = height * area

    if leaf_volume is None:
        leaf_volume = np.zeros(tree.num_leaves(), dtype=np.float64)

    return hg.accumulate_and_add_sequential(tree, height, leaf_volume, hg.Accumulators.sum)


@_legacy_cache_keywords
@hg.argument_helper(hg.CptHierarchy)
def attribute_lca_map(tree, leaf_graph):
    """
    Compute the lowest common ancestor of `i` and `j` for each edge :math:`(i, j)` of the leaf graph of the given tree.

    Complexity: 
    
    The runtime complexity is `\mathcal{O}(m)` where :math:`m` is the number of edges in :attr:`leaf_graph` plus :math:`\mathcal{O}(n\log(n))` on the first call (pre-processing) where :math:`n` is the number of nodes in `tree`.

    :Example:
    =======

    Consider the following 3 x 3 grid graph, whose edges are labelled with their weights:

    .. code-block:: text

        (0)-- 0 --(1)-- 2 --(2)
        |         |         |
        6         6         0
        |         |         |
        (3)-- 0 --(4)-- 4 --(5)
        |         |         |
        5         5         3
        |         |         |
        (6)-- 0 --(7)-- 1 --(8)

    and the binary partition tree computed on it with :func:`~higra.bpt_canonical`:

    .. code-block:: text

        4                 +-------16------+
                          |               |
        3         +-------15-----+        |
                  |              |        |
        2     +---14--+          |        |
              |       |          |        |
        1     |       |       +--13-+     |
              |       |       |     |     |
        0   +-9-+   +-10+   +-12+   |   +-11+
            +   +   +   +   +   +   +   +   +
            0   1   2   5   6   7   8   3   4

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0, 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1))

        tree, altitudes = hg.bpt_canonical(graph, edge_weights)
        hg.attribute_lca_map(tree)
        # array([ 9, 16, 14, 16, 10, 11, 16, 16, 16, 15, 12, 13])

    The result contains one value per edge of the graph, in the order of the edges of `graph`.
    For example:

    - the edge :math:`\{0, 1\}` (first edge): the leaves 0 and 1 are merged in the node 9;
    - the edge :math:`\{1, 2\}` (third edge): the leaf 1 is in the node 9 and the leaf 2 in the node 10, which are merged in the node 14;
    - the edge :math:`\{0, 3\}` (second edge): the leaves 0 and 3 are only merged in the root, the node 16.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param leaf_graph: graph on the leaves of the input tree (deduced from :class:`~higra.CptHierarchy` on `tree`)
    :return: a 1d array representing the lowest common ancestor of the two leaves of each edge of `leaf_graph`
    """
    lca = tree.lowest_common_ancestor_preprocess()
    res = lca.lca(leaf_graph)
    return res


@_legacy_cache_keywords
@hg.argument_helper(hg.CptHierarchy)
def attribute_frontier_length(tree, edge_length=None, leaf_graph=None):
    """
    Compute the length of the frontier represented by each node of the given partition tree.

    In a partition tree, each node represent the merging of 2 or more regions.
    The frontier of a node is then defined as the common contour between the merged regions.
    This function compute the length of these common contours as the sum of the length of edges going from one of the
    merged region to the other one

    The lowest common ancestor of the two extremities of each edge of the leaf graph is computed
    (see :func:`~higra.attribute_lca_map`), then the length of each edge is added to the value of its lowest common
    ancestor:

    .. math::

        F(n) = \\sum_{e = \\{i, j\\} \\in E,\\ lca(i, j) = n} l(e)

    where :math:`l(e)` is the length of the edge :math:`e`. As a leaf is never the lowest common ancestor of an edge,
    its value is 0.

    The edge lengths are determined as follows:

    - if :attr:`edge_length` is provided, it gives the length of each edge of the leaf graph (a 1d array with one
      element per edge);
    - otherwise, the length of each edge is given by :func:`~higra.attribute_edge_length` on the leaf graph.

    The result has the same dtype as the edge_length array.

    :Complexity:

    The runtime complexity is :math:`\\mathcal{O}(N\\log(N) + M)` where :math:`N` is the number of nodes in the tree and
    :math:`M` is the number of edges in the leaf graph (computation of the lowest common ancestors followed by one
    accumulation over the edges).

    :Example:
    =======
    
    Consider the following 3 x 3 grid graph, whose edges are labelled with their weights:

    .. code-block:: text

        (0)-- 0 --(1)-- 2 --(2)
        |         |         |
        6         6         0
        |         |         |
        (3)-- 0 --(4)-- 4 --(5)
        |         |         |
        5         5         3
        |         |         |
        (6)-- 0 --(7)-- 1 --(8)

    and the binary partition tree computed on it with :func:`~higra.bpt_canonical`:

    .. code-block:: text

        4                 +-------16------+
                          |               |
        3         +-------15-----+        |
                  |              |        |
        2     +---14--+          |        |
              |       |          |        |
        1     |       |       +--13-+     |
              |       |       |     |     |
        0   +-9-+   +-10+   +-12+   |   +-11+
            +   +   +   +   +   +   +   +   +
            0   1   2   5   6   7   8   3   4

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0, 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1))

        tree, altitudes = hg.bpt_canonical(graph, edge_weights)
        hg.attribute_frontier_length(tree)
        # array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 5.])

    The node 9 merges the leaves 0 and 1: its frontier is the edge :math:`\\{0, 1\\}`, hence a length of 1.
    The root merges :math:`\\{0, 1, 2, 5, 6, 7, 8\\}` and :math:`\\{3, 4\\}`: its frontier is made of the edges
    :math:`\\{0, 3\\}`, :math:`\\{1, 4\\}`, :math:`\\{3, 6\\}`, :math:`\\{4, 5\\}` and :math:`\\{4, 7\\}`, hence a length of 5.

    With custom edge lengths, here the edge :math:`e_k` has length :math:`k+1`:

    .. code-block:: python

        edge_length = np.arange(1, 13)
        hg.attribute_frontier_length(tree, edge_length=edge_length)
        # array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  5,  6, 11, 12,  3, 10, 30])

    The frontier of the root is now the sum of the lengths of the edges :math:`e_1, e_3, e_6, e_7, e_8`, that is
    :math:`2 + 4 + 7 + 8 + 9 = 30`.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param edge_length: length of the edges of the leaf graph (provided by :func:`~higra.attribute_edge_length` on `leaf_graph`)
    :param leaf_graph: graph on the leaves of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :return: a 1d array representing the length of the frontier of each node of the tree
    """

    if edge_length is None:
        edge_length = hg.attribute_edge_length(leaf_graph)

    lca_map = attribute_lca_map(tree, leaf_graph)

    frontier_length = np.zeros((tree.num_vertices(),), dtype=edge_length.dtype)
    np.add.at(frontier_length, lca_map, edge_length)
    return frontier_length


@_legacy_cache_keywords
@hg.argument_helper(hg.CptHierarchy)
def attribute_frontier_strength(tree, edge_weights, leaf_graph):
    """
    Compute the mean edge weight along the frontier represented by each node of the given partition tree.

    In a partition tree, each node represent the merging of 2 or more regions.
    The frontier of a node is then defined as the common contour between the merged regions.
    This function compute the strength of a common contour as the sum of the weights of edges going from one of the
    merged region to the other one divided by the length of the contour.

    The strength of a frontier is the sum of its edge weights divided by its
    length. The weights are accumulated on the edges of the leaf graph when
    they are provided on the underlying graph of a region adjacency graph.
    The frontier length is computed using the default edge lengths of the
    leaf graph (see :func:`~higra.attribute_frontier_length`).

    Let :math:`E` be the edge set of the leaf graph, and let
    :math:`E_n = \{e = \{i,j\} \in E \mid lca(i,j)=n\}`. Then:

    .. math::

        S(n) = \frac{\sum_{e \in E_n} w(e)}{\sum_{e \in E_n} l(e)}

    where :math:`w(e)` is the edge weight, after any required accumulation
    from the underlying graph, and :math:`l(e)` is the default edge length
    used by :func:`~higra.attribute_frontier_length`. For an ordinary graph
    with unit edge lengths, the denominator is the number of frontier edges.
    For a region adjacency graph, it is the total length of the frontier.
    The value of a leaf is 0.

    If the leaf graph is a region adjacency graph and `edge_weights` has one value per edge of its underlying graph,
    the weights are first summed on the edges of the region adjacency graph (see :func:`~higra.rag_accumulate_on_edges`).

    The result has the same dtype as the edge_weights array: with integer weights, the mean is truncated.

    :Complexity:

    The runtime complexity is :math:`\\mathcal{O}(N\\log(N) + M)` where :math:`N` is the number of nodes in the tree and
    :math:`M` is the number of edges in the leaf graph.

    :Example:
    =======

    Consider the following 3 x 3 grid graph, whose edges are labelled with their weights:

    .. code-block:: text

        (0)-- 0 --(1)-- 2 --(2)
        |         |         |
        6         6         0
        |         |         |
        (3)-- 0 --(4)-- 4 --(5)
        |         |         |
        5         5         3
        |         |         |
        (6)-- 0 --(7)-- 1 --(8)

    and the binary partition tree computed on it with :func:`~higra.bpt_canonical`:

    .. code-block:: text

        4                 +-------16------+
                          |               |
        3         +-------15-----+        |
                  |              |        |
        2     +---14--+          |        |
              |       |          |        |
        1     |       |       +--13-+     |
              |       |       |     |     |
        0   +-9-+   +-10+   +-12+   |   +-11+
            +   +   +   +   +   +   +   +   +
            0   1   2   5   6   7   8   3   4

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0., 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1))

        tree, altitudes = hg.bpt_canonical(graph, edge_weights)
        hg.attribute_frontier_strength(tree, edge_weights)
        # array([0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. , 2. , 3. , 5.2])

    The frontier of the node 14 is the edge :math:`\\{1, 2\\}` of weight 2, the one of the node 15 is the edge
    :math:`\\{5, 8\\}` of weight 3. The frontier of the root is made of the edges :math:`\\{0, 3\\}`,
    :math:`\\{1, 4\\}`, :math:`\\{3, 6\\}`, :math:`\\{4, 5\\}` and :math:`\\{4, 7\\}` of weights 6, 6, 5, 4 and 5:
    its strength is :math:`(6 + 6 + 5 + 4 + 5) / 5 = 5.2`.

    With integer weights, the result is also an integer array and the strength of the root is truncated to 5.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param edge_weights: weight of the edges of the leaf graph (if leaf_graph is a region adjacency graph, edge_weights might be weights on the edges of the pre-graph of the rag). (provided by :func:`~higra.rag_accumulate_on_edges` on `leaf_graph`)
    :param leaf_graph: graph on the leaves of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :return: a 1d array representing the mean edge weight along the frontier of each node of the tree
    """
    # this is a rag like graph
    if hg.CptRegionAdjacencyGraph.validate(leaf_graph) and edge_weights.shape[0] != leaf_graph.num_edges():
        edge_weights = hg.rag_accumulate_on_edges(leaf_graph, hg.Accumulators.sum, edge_weights=edge_weights)

    frontier_length = hg.attribute_frontier_length(tree, leaf_graph=leaf_graph)
    frontier_strength = hg.attribute_frontier_length(tree, edge_weights, leaf_graph)
    frontier_strength[tree.num_leaves():] = frontier_strength[tree.num_leaves():] / frontier_length[tree.num_leaves():]
    return frontier_strength


@_legacy_cache_keywords
@hg.argument_helper(hg.CptHierarchy)
def attribute_contour_length(tree, vertex_perimeter=None, edge_length=None, leaf_graph=None):
    """
    Compute the length of the contour (perimeter) of each node of the given tree.

    Each leaf :math:`v` of the tree is a vertex of the leaf graph (for example a pixel) with a perimeter
    :math:`p(v)`, and each edge :math:`e` of the leaf graph has a length :math:`l(e)`: the length of the border shared
    by the two regions it connects. The contour of a node :math:`n` is the boundary of the region made of the leaves
    of the subtree rooted in :math:`n`. Summing the perimeters of its leaves counts every border shared by two
    leaves of :math:`n` twice: these inner borders must be removed to obtain the contour. Every inner border
    is the frontier of a node of the subtree rooted in :math:`n` (see :func:`~higra.attribute_frontier_length`), hence:

    .. math::

        C(n) = \\sum_{v \\in leaves(n)} p(v) \\; - \\; 2 \\sum_{m \\in subtree(n)} F(m)

    where :math:`subtree(n)` is the set of the non-leaf nodes of the subtree rooted in :math:`n` (including :math:`n`)
    and :math:`F(m)` is the frontier length of the node :math:`m`:

    .. math::

        F(m) = \\sum_{e = \\{i, j\\} \\in E,\\ lca(i, j) = m} l(e)

    where :math:`E` is the edge set of the leaf graph. The contour of a leaf is its own perimeter.

    The perimeters and lengths are determined as follows:

    - if :attr:`vertex_perimeter` is provided, it gives the perimeter of each leaf: a 1d array with one element per
      leaf or, if the leaf graph is a grid graph (e.g. an image), an array whose first dimensions follow the shape
      of the grid; otherwise the perimeters are given by :func:`~higra.attribute_vertex_perimeter` on the leaf graph;
    - if :attr:`edge_length` is provided, it gives the length of each edge of the leaf graph (a 1d array with one
      element per edge); otherwise the lengths are given by :func:`~higra.attribute_edge_length` on the leaf graph.

    The borders of the domain (for example the sides of the image) belong to the contour of the nodes touching them.

    :Complexity:

    The runtime complexity is :math:`\\mathcal{O}(N\\log(N) + M)` where :math:`N` is the number of nodes in the tree and
    :math:`M` is the number of edges in the leaf graph (frontier lengths followed by one accumulation over the tree).
    If :attr:`vertex_perimeter` or :attr:`edge_length` are omitted, add the cost of their computation.

    :Example:
    =======

    Consider the following 3 x 3 grid graph, whose edges are labelled with their weights:

    .. code-block:: text

        (0)-- 0 --(1)-- 2 --(2)
        |         |         |
        6         6         0
        |         |         |
        (3)-- 0 --(4)-- 4 --(5)
        |         |         |
        5         5         3
        |         |         |
        (6)-- 0 --(7)-- 1 --(8)

    and the binary partition tree computed on it with :func:`~higra.bpt_canonical`:

    .. code-block:: text

        4                 +-------16------+
                          |               |
        3         +-------15-----+        |
                  |              |        |
        2     +---14--+          |        |
              |       |          |        |
        1     |       |       +--13-+     |
              |       |       |     |     |
        0   +-9-+   +-10+   +-12+   |   +-11+
            +   +   +   +   +   +   +   +   +
            0   1   2   5   6   7   8   3   4

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0., 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1))

        tree, altitudes = hg.bpt_canonical(graph, edge_weights)
        hg.attribute_contour_length(tree)
        # array([ 4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  6.,  6.,  6.,  6.,  8., 10., 16., 12.])

    Each pixel has a perimeter of 4. The node 9 is the region :math:`\\{0, 1\\}` (two adjacent pixels): its contour is
    :math:`4 + 4 - 2 \\times 1 = 6`. The node 13 is the bottom row :math:`\\{6, 7, 8\\}`:
    :math:`3 \\times 4 - 2 \\times 2 = 8`. The root is the whole image: :math:`9 \\times 4 - 2 \\times 12 = 12`, the
    perimeter of the 3 x 3 square.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param vertex_perimeter: perimeter of each vertex of the leaf graph (provided by :func:`~higra.attribute_vertex_perimeter` on `leaf_graph`)
    :param edge_length: length of each edge of the leaf graph (provided by :func:`~higra.attribute_edge_length` on `leaf_graph`)
    :param leaf_graph: graph on the leaves of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :return: a 1d array representing the contour length of each node of the tree
    """

    if vertex_perimeter is None:
        vertex_perimeter = hg.attribute_vertex_perimeter(leaf_graph)

    if edge_length is None:
        edge_length = hg.attribute_edge_length(leaf_graph)

    if leaf_graph is not None:
        vertex_perimeter = hg.linearize_vertex_weights(vertex_perimeter, leaf_graph)

    frontier_length = hg.attribute_frontier_length(tree, edge_length, leaf_graph)
    perimeter = hg.accumulate_and_add_sequential(tree, -2 * frontier_length, vertex_perimeter,
                                                 hg.Accumulators.sum)

    # hg.cpp._attribute_contour_length_component_tree is more efficient than the partition tree
    # algorithm but it does not work for tree of shapes left in original space (the problem is that
    # two children of a node may become adjacent when the interpolated pixels are removed).

    # if tree.category() == hg.TreeCategory.PartitionTree:
    #     frontier_length = hg.attribute_frontier_length(tree, edge_length, leaf_graph)
    #     perimeter = hg.accumulate_and_add_sequential(tree, -2 * frontier_length, vertex_perimeter,
    #                                                  hg.Accumulators.sum)
    # elif tree.category() == hg.TreeCategory.ComponentTree:
    #     perimeter = hg.cpp._attribute_contour_length_component_tree(tree, leaf_graph, vertex_perimeter,
    #                                                                 edge_length)

    return perimeter


@_legacy_cache_keywords
@hg.argument_helper(hg.CptHierarchy)
def attribute_contour_strength(tree, edge_weights, vertex_perimeter=None, edge_length=None, leaf_graph=None):
    """
    Compute the strength of the contour of each node of the given tree.

    The strength of the contour of a node :math:`n` is the total weight of the edges of the leaf graph crossing the
    contour of :math:`n`, divided by the length of the contour (see :func:`~higra.attribute_contour_length`):

    .. math::

        S(n) = \\frac{W(n)}{C(n)}
        \\qquad\\text{with}\\qquad
        W(n) = \\sum_{e = \\{i, j\\} \\in E,\\ |\\{i, j\\} \\cap leaves(n)| = 1} w(e)

    where :math:`E` is the edge set of the leaf graph, :math:`w(e)` is the weight of the edge :math:`e`, and
    :math:`C(n)` is the contour length of :math:`n`. The edges considered in :math:`W(n)` are the edges having exactly
    one end in :math:`n`. The weight of an edge is counted for its whole length, and the borders of the domain
    (for example the sides of an image) have no weight: they only contribute to :math:`C(n)`. If the contour length
    of the root is null, it is replaced by 1.

    If the leaf graph is a region adjacency graph, :attr:`edge_weights` gives one weight per edge of the underlying
    graph of the region adjacency graph: the weights are first summed on the edges of the region adjacency graph
    (see :func:`~higra.rag_accumulate_on_edges`).

    The perimeters and lengths are determined as in :func:`~higra.attribute_contour_length`.

    :Complexity:

    The runtime complexity is :math:`\\mathcal{O}(N\\log(N) + M)` where :math:`N` is the number of nodes in the tree and
    :math:`M` is the number of edges in the leaf graph (two contour computations).
    If :attr:`vertex_perimeter` or :attr:`edge_length` are omitted, add the cost of their computation.

    :Example:
    =======

    Consider the following 3 x 3 grid graph, whose edges are labelled with their weights:

    .. code-block:: text

        (0)-- 0 --(1)-- 2 --(2)
        |         |         |
        6         6         0
        |         |         |
        (3)-- 0 --(4)-- 4 --(5)
        |         |         |
        5         5         3
        |         |         |
        (6)-- 0 --(7)-- 1 --(8)

    and the binary partition tree computed on it with :func:`~higra.bpt_canonical`:

    .. code-block:: text

        4                 +-------16------+
                          |               |
        3         +-------15-----+        |
                  |              |        |
        2     +---14--+          |        |
              |       |          |        |
        1     |       |       +--13-+     |
              |       |       |     |     |
        0   +-9-+   +-10+   +-12+   |   +-11+
            +   +   +   +   +   +   +   +   +
            0   1   2   5   6   7   8   3   4

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0., 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1))

        tree, altitudes = hg.bpt_canonical(graph, edge_weights)
        hg.attribute_contour_strength(tree, edge_weights)
        # array([1.5, 2. , 0.5, 2.75, 3.75, 1.75, 1.25, 1.5, 1., 2.33333333, 1.5, 4.33333333, 1.83333333,
        #        1.625, 1.9, 1.625, 0.])

    The node 9 is the region :math:`\\{0, 1\\}`: the edges leaving it are :math:`\\{0, 3\\}`, :math:`\\{1, 4\\}` and
    :math:`\\{1, 2\\}` of weights 6, 6 and 2, and its contour length is 6, hence a strength of
    :math:`(6 + 6 + 2) / 6 = 2.33`. The node 13 is the bottom row :math:`\\{6, 7, 8\\}`: the leaving edges have weights
    5, 5 and 3 and the contour length is 8, hence :math:`13 / 8 = 1.625`. The root has no leaving edge: its strength is 0.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param edge_weights: weight of the edges of the leaf graph (if `leaf_graph` is a region adjacency graph, weights of the edges of its underlying graph)
    :param vertex_perimeter: perimeter of each vertex of the leaf graph (provided by :func:`~higra.attribute_vertex_perimeter` on `leaf_graph`)
    :param edge_length: length of each edge of the leaf graph (provided by :func:`~higra.attribute_edge_length` on `leaf_graph`)
    :param leaf_graph: graph on the leaves of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :return: a 1d array representing the strength of the contour of each node of the tree
    """

    if vertex_perimeter is None:
        vertex_perimeter = hg.attribute_vertex_perimeter(leaf_graph)

    if edge_length is None:
        edge_length = hg.attribute_edge_length(leaf_graph)

    perimeter = attribute_contour_length(tree, vertex_perimeter, edge_length, leaf_graph)

    # perimeter of the root may be null
    if np.isclose(perimeter[-1], 0):
        perimeter[-1] = 1

    if hg.CptRegionAdjacencyGraph.validate(leaf_graph):
        edge_weights = hg.rag_accumulate_on_edges(leaf_graph, hg.Accumulators.sum, edge_weights)

    vertex_weights_sum = hg.accumulate_graph_edges(leaf_graph, edge_weights, hg.Accumulators.sum)
    edge_weights_sum = attribute_contour_length(tree, vertex_weights_sum, edge_weights, leaf_graph)

    return edge_weights_sum / perimeter


@_legacy_cache_keywords
@hg.argument_helper(hg.CptHierarchy)
def attribute_compactness(tree, area=None, contour_length=None, normalize=True, leaf_graph=None):
    """
    Compute the compactness of each node of the given tree.

    The compactness of a node :math:`n` is the ratio between its area and the square of the length of its contour:

    .. math::

        K(n) = \\frac{A(n)}{C(n)^2}

    where :math:`A(n)` is the area of :math:`n` (see :func:`~higra.attribute_area`) and :math:`C(n)` is the length of
    its contour (see :func:`~higra.attribute_contour_length`). For a given area, the more compact the region, the
    shorter its contour and the larger its compactness.

    If :attr:`normalize` is ``True``, the values are divided by the largest value of the array, so that the most compact
    node of the tree has a compactness of 1:

    .. math::

        K_{norm}(n) = \\frac{K(n)}{\\max_m K(m)}

    The area and the contour length are determined as follows:

    - if :attr:`area` (resp. :attr:`contour_length`) is provided, it is used as such;
    - otherwise, it is computed with :func:`~higra.attribute_area` (resp. :func:`~higra.attribute_contour_length`).

    :Complexity:

    The computation takes :math:`\\mathcal{O}(N)` time, where :math:`N` is the number of nodes in the tree.
    If :attr:`area` or :attr:`contour_length` are omitted, add the cost of :func:`~higra.attribute_area` and
    :func:`~higra.attribute_contour_length`.

    :Example:
    =======

    Consider the following 3 x 3 grid graph, whose edges are labelled with their weights:

    .. code-block:: text

        (0)-- 0 --(1)-- 2 --(2)
        |         |         |
        6         6         0
        |         |         |
        (3)-- 0 --(4)-- 4 --(5)
        |         |         |
        5         5         3
        |         |         |
        (6)-- 0 --(7)-- 1 --(8)

    and the binary partition tree computed on it with :func:`~higra.bpt_canonical`:

    .. code-block:: text

        4                 +-------16------+
                          |               |
        3         +-------15-----+        |
                  |              |        |
        2     +---14--+          |        |
              |       |          |        |
        1     |       |       +--13-+     |
              |       |       |     |     |
        0   +-9-+   +-10+   +-12+   |   +-11+
            +   +   +   +   +   +   +   +   +
            0   1   2   5   6   7   8   3   4

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0., 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1))

        tree, altitudes = hg.bpt_canonical(graph, edge_weights)
        hg.attribute_compactness(tree, normalize=False)
        # array([0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.05555556,
        #        0.05555556, 0.05555556, 0.05555556, 0.046875, 0.04, 0.02734375, 0.0625])

    A pixel has an area of 1 and a contour of 4: :math:`1 / 4^2 = 0.0625`. The node 9 has an area of 2 and a contour of 6:
    :math:`2 / 6^2 = 0.0556`. The root has an area of 9 and a contour of 12: :math:`9 / 12^2 = 0.0625`.

    With the default normalization, the largest value (0.0625, reached by the leaves and the root) becomes 1:

    .. code-block:: python

        hg.attribute_compactness(tree)
        # array([1., 1., 1., 1., 1., 1., 1., 1., 1., 0.88888889, 0.88888889, 0.88888889, 0.88888889,
        #        0.75, 0.64, 0.4375, 1.])

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param area: node area of the input tree (provided by :func:`~higra.attribute_area` on `tree`)
    :param contour_length: node contour length of the input tree (provided by :func:`~higra.attribute_contour_length` on `tree`)
    :param normalize: if True the result is divided by the maximal compactness value in the tree
    :param leaf_graph: graph on the leaves of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :return: a 1d array representing the compactness of each node of the tree
    """
    if area is None:
        area = hg.attribute_area(tree)

    if contour_length is None:
        contour_length = hg.attribute_contour_length(tree, leaf_graph=leaf_graph)

    compactness = area / (contour_length * contour_length)
    if normalize:
        max_compactness = np.nanmax(compactness)
        compactness = compactness / max_compactness

    return compactness


@_legacy_cache_keywords
@hg.argument_helper(hg.CptHierarchy)
def attribute_mean_vertex_weights(tree, vertex_weights, area=None, leaf_graph=None):
    """
    Compute the mean of the vertex weights of the leaf graph vertices inside each node of the given tree.

    Each leaf of the tree is a vertex of the leaf graph (for example a pixel) with a weight :math:`w(v)` (for example
    its gray level, or a vector such as its color). For any node :math:`n`, the mean vertex weight :math:`M(n)` is the
    sum of the weights of the leaves of the subtree rooted in :math:`n`, divided by the area of :math:`n`:

    .. math::

        M(n) = \\frac{\\sum_{v \\in leaves(n)} w(v)}{A(n)}

    where :math:`A(n)` is the area of :math:`n` (see :func:`~higra.attribute_area`). The mean of a leaf is its own
    weight.

    The weights may be scalars or vectors: for vector weights of dimension :math:`d`, the mean is computed independently
    for each component. The weights are given as follows:

    - :attr:`vertex_weights` is an array with one element per leaf or, if the leaf graph is a grid graph (e.g. an image),
      an array whose first dimensions follow the shape of the grid;
    - the area of the nodes is :attr:`area` if provided, otherwise it is computed with :func:`~higra.attribute_area`.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(N \\times d)` time, where :math:`N` is the number of nodes in the tree and
    :math:`d` is the dimension of the vertex weights (:math:`d = 1` for scalar weights). If :attr:`area` is omitted,
    add the cost of :func:`~higra.attribute_area`.

    :Example:
    =======

    Consider the following 3 x 3 grid graph, whose edges are labelled with their weights:

    .. code-block:: text

        (0)-- 0 --(1)-- 2 --(2)
        |         |         |
        6         6         0
        |         |         |
        (3)-- 0 --(4)-- 4 --(5)
        |         |         |
        5         5         3
        |         |         |
        (6)-- 0 --(7)-- 1 --(8)

    and the binary partition tree computed on it with :func:`~higra.bpt_canonical`:

    .. code-block:: text

        4                 +-------16------+
                          |               |
        3         +-------15-----+        |
                  |              |        |
        2     +---14--+          |        |
              |       |          |        |
        1     |       |       +--13-+     |
              |       |       |     |     |
        0   +-9-+   +-10+   +-12+   |   +-11+
            +   +   +   +   +   +   +   +   +
            0   1   2   5   6   7   8   3   4

    The vertex weights are the gray levels of the 3 x 3 image:

    .. code-block:: text

        1  2  3
        4  5  6
        7  8  9

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0, 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1))
        vertex_weights = np.asarray(((1., 2, 3), (4, 5, 6), (7, 8, 9)))

        tree, altitudes = hg.bpt_canonical(graph, edge_weights)
        hg.attribute_mean_vertex_weights(tree, vertex_weights)
        # array([1.        , 2.        , 3.        , 4.        , 5.        ,
        #        6.        , 7.        , 8.        , 9.        , 1.5       ,
        #        4.5       , 4.5       , 7.5       , 8.        , 3.        ,
        #        5.14285714, 5.        ])

    The node 9 contains the pixels 0 and 1: its mean is :math:`(1 + 2) / 2 = 1.5`.
    The node 13 contains the pixels 6, 7 and 8: its mean is :math:`(7 + 8 + 9) / 3 = 8`.
    The node 14 contains the pixels 0, 1, 2 and 5: its mean is :math:`(1 + 2 + 3 + 6) / 4 = 3`.
    The root is the mean of all the weights: :math:`45 / 9 = 5`.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param vertex_weights: vertex weights of the leaf graph of the input tree (scalar or vectorial)
    :param area: area of the tree nodes (provided by :func:`~higra.attribute_area` on `tree`)
    :param leaf_graph: leaf graph of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :return: a nd array of floating point values with one element per node of the tree (and the same trailing dimension as :attr:`vertex_weights` if the weights are vectorial)
    """
    if area is None:
        area = hg.attribute_area(tree)

    if leaf_graph is not None:
        vertex_weights = hg.linearize_vertex_weights(vertex_weights, leaf_graph)

    attribute = hg.accumulate_sequential(
        tree,
        vertex_weights.astype(np.float64),
        hg.Accumulators.sum) / area.reshape([-1] + [1] * (vertex_weights.ndim - 1))
    return attribute


@_legacy_cache_keywords
def attribute_sibling(tree, skip=1):
    """
    Compute the sibling of each node of the given tree.

    The children of a node :math:`p` are ordered. Let :math:`n` be the :math:`k`-th child of :math:`p` (starting from
    0) among the :math:`N_p` children of :math:`p`. The sibling of :math:`n` is the node of index:

    .. math::

        sibling(n) = child\\left(p,\\ (k + skip) \\bmod N_p\\right)

    where :math:`child(p, i)` is the :math:`i`-th child of :math:`p`. The children of a node are thus visited in
    a circular way: with :attr:`skip` equal to 1, the sibling of a node is the next child of its parent, and the sibling of
    the last child is the first one. The sibling of the root, which has no parent, is itself.

    The sibling attribute makes it possible to emulate a (circular, doubly) linked list among the children of a node.
    In a binary tree, with :attr:`skip` equal to 1, the sibling of a node is its only brother.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(N)` time, where :math:`N` is the number of nodes in the tree.

    :Example:
    =======

    Consider the following binary tree with 5 leaves:

    .. code-block:: text

        3         +-----8-----+
                  |           |
        2     +---7---+       |
              |       |       |
        1   +-5-+   +-6-+     |
            |   |   |   |     |
        0   0   1   2   3     4

    .. code-block:: python

        tree = hg.Tree((5, 5, 6, 6, 8, 7, 7, 8, 8))
        hg.attribute_sibling(tree)
        # array([1, 0, 3, 2, 7, 6, 5, 4, 8])

    The sibling of each node is its only brother: the leaves 0 and 1 are siblings, as are the leaves 2 and 3, the nodes 5 and 6,
    and the node 4 and the node 7. The root is its own sibling.

    With :attr:`skip` equal to 2:

    .. code-block:: python

        hg.attribute_sibling(tree, skip=2)
        # array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    Each node has a single brother, so skipping 2 positions in the circular list of children brings back to the node itself.

    :param tree: input tree
    :param skip: number of positions to skip in the list of children (including the node itself): 1 for the next child
    :return: a 1d array of node indices representing the sibling of each node of the tree
    """
    attribute = hg.cpp._attribute_sibling(tree, skip)
    return attribute


@_legacy_cache_keywords
def attribute_depth(tree):
    """
    Compute the depth of each node of the given tree.

    The depth :math:`d(n)` of a node :math:`n` is the number of ancestors of :math:`n`, that is the number of edges on the
    path from :math:`n` to the root:

    .. math::

        d(n) = \\begin{cases}
            0 & \\textrm{if } n \\textrm{ is the root}\\\\
            d(parent(n)) + 1 & \\textrm{otherwise}
        \\end{cases}

    :Complexity:

    The computation takes :math:`\\mathcal{O}(N)` time, where :math:`N` is the number of nodes in the tree.

    :Example:
    =======

    Consider the following binary tree with 5 leaves:

    .. code-block:: text

        3         +-----8-----+
                  |           |
        2     +---7---+       |
              |       |       |
        1   +-5-+   +-6-+     |
            |   |   |   |     |
        0   0   1   2   3     4

    .. code-block:: python

        tree = hg.Tree((5, 5, 6, 6, 8, 7, 7, 8, 8))
        hg.attribute_depth(tree)
        # array([3, 3, 3, 3, 1, 2, 2, 1, 0])

    The root has a depth of 0, the node 7 and the leaf 4 are its children (depth 1), and the leaves 0 to 3 are at
    depth 3: their ancestors are the nodes 5 or 6, the node 7 and the root.

    :param tree: input tree
    :return: a 1d array representing the depth of each node of the tree
    """
    attribute = hg.cpp._attribute_depth(tree)
    return attribute


@_legacy_cache_keywords
def attribute_regular_altitudes(tree, depth=None):
    """
    Compute regular altitudes for the nodes of the given tree, derived from their depth.

    The regular altitude :math:`alt(n)` of a node :math:`n` decreases linearly with its depth :math:`d(n)` (see
    :func:`~higra.attribute_depth`), from 1 for the root to 0 for the nodes at the largest depth, and the leaves are
    always at altitude 0:

    .. math::

        alt(n) = \\begin{cases}
            0 & \\textrm{if } n \\textrm{ is a leaf}\\\\
            1 - \\frac{d(n)}{\\max_m d(m)} & \\textrm{otherwise}
        \\end{cases}

    The altitudes are thus comprised between 0 and 1 and are increasing from the leaves to the root.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(N)` time, where :math:`N` is the number of nodes in the tree.
    If :attr:`depth` is omitted, add the cost of :func:`~higra.attribute_depth`.

    :Example:
    =======

    Consider the following binary tree with 5 leaves:

    .. code-block:: text

        3         +-----8-----+
                  |           |
        2     +---7---+       |
              |       |       |
        1   +-5-+   +-6-+     |
            |   |   |   |     |
        0   0   1   2   3     4

    .. code-block:: python

        tree = hg.Tree((5, 5, 6, 6, 8, 7, 7, 8, 8))
        hg.attribute_regular_altitudes(tree)
        # array([0.        , 0.        , 0.        , 0.        , 0.        , 0.33333333,
        #        0.33333333, 0.66666667, 1.        ])

    The depths of the nodes 5 to 8 are 2, 2, 1 and 0 and the largest depth is 3: the altitudes are
    :math:`1 - 2/3`, :math:`1 - 2/3`, :math:`1 - 1/3` and :math:`1 - 0/3`.

    :param tree: input tree
    :param depth: depth of the tree nodes (provided by :func:`~higra.attribute_depth` on `tree`)
    :return: a 1d array of floating point values representing the regular altitude of each node of the tree
    """

    if depth is None:
        depth = hg.attribute_depth(tree)

    altitudes = 1 - depth / np.max(depth)
    altitudes[:tree.num_leaves()] = 0
    return altitudes


@_legacy_cache_keywords
def attribute_vertex_list(tree):
    """
    List the leaves of the subtree rooted in each node of the given tree.

    For a node :math:`n`, the result contains the list of the leaves :math:`leaves(n)` of the subtree rooted in
    :math:`n`. The list of a leaf contains only the leaf itself, and the list of an internal node is the concatenation of
    the lists of its children, so the list of the root contains all the leaves.

    **WARNING**: this function is slow and uses :math:`\\mathcal{O}(n^2)` memory in the worst case, with :math:`n`
    the number of leaves of the tree. It should only be used for debugging and testing.

    :Complexity:

    The runtime and memory complexity is :math:`\\mathcal{O}(\\sum_{m} |leaves(m)|)` (the total size of the lists),
    which is :math:`\\mathcal{O}(N \\times n)` in the worst case, where :math:`N` is the number of nodes
    and :math:`n` the number of leaves of the tree.

    :Example:
    =======

    Consider the following binary tree with 5 leaves:

    .. code-block:: text

        3         +-----8-----+
                  |           |
        2     +---7---+       |
              |       |       |
        1   +-5-+   +-6-+     |
            |   |   |   |     |
        0   0   1   2   3     4

    .. code-block:: python

        tree = hg.Tree((5, 5, 6, 6, 8, 7, 7, 8, 8))
        hg.attribute_vertex_list(tree)
        # [[0], [1], [2], [3], [4], [0, 1], [2, 3], [0, 1, 2, 3], [4, 0, 1, 2, 3]]

    The list of the node 7 is the concatenation of the lists of the nodes 5 and 6.

    :param tree: input tree
    :return: a list with one element per node of the tree, each element being the list of the leaves of the subtree rooted in this node
    """
    result = [[i] for i in tree.leaves()]

    for i in tree.leaves_to_root_iterator(include_leaves=False):
        tmp = []
        for c in tree.children(i):
            tmp.extend(result[c])
        result.append(tmp)

    return result

@_legacy_cache_keywords
@hg.argument_helper(hg.CptHierarchy)
def attribute_gaussian_region_weights_model(tree, vertex_weights, leaf_graph=None):
    """
    Estimate a gaussian model (mean and variance or covariance) of the vertex weights inside each node of the given tree.

    Each leaf of the tree is a vertex of the leaf graph (for example a pixel) with a weight :math:`w(v)`, scalar or
    vectorial (for example a gray level or a color). For a node :math:`n` of area :math:`A(n)` (see
    :func:`~higra.attribute_area`), the model is made of:

    - the mean :math:`\\mu(n)` of the weights of the leaves of :math:`n`:

      .. math::

          \\mu(n) = \\frac{1}{A(n)} \\sum_{v \\in leaves(n)} w(v)

    - the (biased) variance :math:`\\sigma^2(n)` of the weights if they are scalar, or the (biased) covariance matrix
      :math:`\\Sigma(n)` of the weights if they are vectors:

      .. math::

          \\sigma^2(n) = \\frac{1}{A(n)} \\sum_{v \\in leaves(n)} w(v)^2 - \\mu(n)^2
          \\qquad\\qquad
          \\Sigma(n) = \\frac{1}{A(n)} \\sum_{v \\in leaves(n)} w(v)\\, w(v)^{T} - \\mu(n)\\, \\mu(n)^{T}

    The mean and the variance of a leaf are respectively its weight and 0 (a null matrix for vector weights).

    The vertex weights must be scalar or 1 dimensional vectors: :attr:`vertex_weights` is an array with one element per
    leaf (of shape :math:`(n,)` or :math:`(n, d)`) or, if the leaf graph is a grid graph (e.g. an image), an array
    whose first dimensions follow the shape of the grid.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(N)` time for scalar weights and :math:`\\mathcal{O}(N \\times d^2)` time
    for vector weights of dimension :math:`d`, where :math:`N` is the number of nodes in the tree, plus the cost of
    :func:`~higra.attribute_area`.

    :Example:
    =======

    Consider the following 3 x 3 grid graph, whose edges are labelled with their weights:

    .. code-block:: text

        (0)-- 0 --(1)-- 2 --(2)
        |         |         |
        6         6         0
        |         |         |
        (3)-- 0 --(4)-- 4 --(5)
        |         |         |
        5         5         3
        |         |         |
        (6)-- 0 --(7)-- 1 --(8)

    and the binary partition tree computed on it with :func:`~higra.bpt_canonical`:

    .. code-block:: text

        4                 +-------16------+
                          |               |
        3         +-------15-----+        |
                  |              |        |
        2     +---14--+          |        |
              |       |          |        |
        1     |       |       +--13-+     |
              |       |       |     |     |
        0   +-9-+   +-10+   +-12+   |   +-11+
            +   +   +   +   +   +   +   +   +
            0   1   2   5   6   7   8   3   4

    The vertex weights are the gray levels of the 3 x 3 image:

    .. code-block:: text

        1  2  3
        4  5  6
        7  8  9

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0, 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1))
        vertex_weights = np.asarray(((1., 2, 3), (4, 5, 6), (7, 8, 9)))

        tree, altitudes = hg.bpt_canonical(graph, edge_weights)
        mean, variance = hg.attribute_gaussian_region_weights_model(tree, vertex_weights)
        # mean:     array([1.  , 2.  , 3.  , 4.  , 5.  , 6.  , 7.  , 8.  , 9.  , 1.5 , 4.5 , 4.5 , 7.5 , 8.  , 3.  ,
        #                  5.14285714, 5.  ])
        # variance: array([0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 2.25, 0.25, 0.25, 0.66666667,
        #                  3.5 , 8.40816327, 6.66666667])

    The node 14 contains the pixels 0, 1, 2 and 5, whose weights are :math:`\\{1, 2, 3, 6\\}`: their mean is 3 and their variance is
    :math:`((1 - 3)^2 + (2 - 3)^2 + (3 - 3)^2 + (6 - 3)^2) / 4 = 3.5`.
    The node 13 contains the weights :math:`\\{7, 8, 9\\}`: their mean is 8 and their variance is :math:`2 / 3`.

    With vector weights of dimension 2, the mean has shape ``(17, 2)`` and the covariance has shape ``(17, 2, 2)``.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param vertex_weights: vertex weights of the leaf graph of the input tree (scalar or 1 dimensional vectors)
    :param leaf_graph: leaf graph of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :return: two arrays ``(mean, variance)``: the mean of each node (scalar or vectorial, as the vertex weights) and the variance of each node (scalar if the weights are scalar, otherwise a covariance matrix)
    """
    if leaf_graph is not None:
        vertex_weights = hg.linearize_vertex_weights(vertex_weights, leaf_graph)

    if vertex_weights.ndim > 2:
        raise ValueError("Vertex weight can either be scalar or 1 dimensional.")

    if vertex_weights.dtype not in (np.float32, np.float64):
        vertex_weights = vertex_weights.astype(np.float64)

    area = hg.attribute_area(tree, leaf_graph=leaf_graph)
    mean = hg.accumulate_sequential(tree, vertex_weights, hg.Accumulators.sum, leaf_graph)

    if vertex_weights.ndim == 1:
        # general case below would work but this is simpler
        mean /= area
        mean2 = hg.accumulate_sequential(tree, vertex_weights * vertex_weights, hg.Accumulators.sum, leaf_graph)
        mean2 /= area
        variance = mean2 - mean * mean
    else:
        mean /= area[:, None]
        tmp = vertex_weights[:, :, None] * vertex_weights[:, None, :]
        mean2 = hg.accumulate_sequential(tree, tmp, hg.Accumulators.sum, leaf_graph)
        mean2 /= area[:, None, None]

        variance = mean2 - mean[:, :, None] * mean[:, None, :]

    return mean, variance


@_legacy_cache_keywords
def attribute_extrema(tree, altitudes):
    """
    Identify the nodes of a hierarchy that represent extrema (minima or maxima).

    An extremum (minimum or maximum) of the hierarchy :math:`T` with altitudes :math:`alt` is a node :math:`n` of :math:`T` such that the
    altitude of any non leaf node included in :math:`n` is equal to the altitude of :math:`n` and the altitude of
    the parent of :math:`n` is different from the altitude of :math:`n`. Formally, :math:`n` is an extremum if:

    .. math::

        \\forall m \\in subtree(n) \\setminus leaves(n),\\ alt(m) = alt(n)
        \\qquad\\textrm{and}\\qquad
        alt(parent(n)) \\neq alt(n)

    The result is a boolean array such that :math:`result(n)` is ``True`` if the node :math:`n` is an extremum and ``False``
    otherwise. Leaves are never extrema. The function does not depend on whether the altitudes are increasing or decreasing:
    the extrema are minima for increasing altitudes and maxima for decreasing altitudes.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(N)` time, where :math:`N` is the number of nodes in the tree.

    :Example:
    =======

    Consider the following binary tree with 5 leaves:

    .. code-block:: text

        3         +-----8-----+
                  |           |
        2     +---7---+       |
              |       |       |
        1   +-5-+   +-6-+     |
            |   |   |   |     |
        0   0   1   2   3     4

    .. code-block:: python

        tree = hg.Tree((5, 5, 6, 6, 8, 7, 7, 8, 8))
        altitudes = np.asarray((0, 0, 0, 0, 0, 1, 2, 4, 5))
        hg.attribute_extrema(tree, altitudes)
        # array([False, False, False, False, False,  True,  True, False, False])

    The nodes 5 and 6 are extrema: they contain no non-leaf node and their altitudes (1 and 2) differ from the
    altitude of their parent (4). The node 7 is not an extremum as it contains the nodes 5 and 6, whose altitudes differ
    from its own.

    :param tree: input tree
    :param altitudes: node altitudes of the input tree (Numpy array of shape (tree.num_vertices(),))
    :return: a 1d boolean array representing, for each node of the tree, whether it is an extremum
    """

    res = hg.cpp._attribute_extrema(tree, altitudes)

    return res


def __process_param_increasing_altitudes(tree, altitudes, increasing_altitudes):
    """
    Determine whether the altitudes of a tree are increasing or decreasing, according to the value of the parameter
    :attr:`increasing_altitudes` of the functions :func:`~higra.attribute_extinction_value`,
    :func:`~higra.attribute_height` and :func:`~higra.attribute_dynamics`.

    The altitudes are assumed to be monotone: either increasing from the leaves to the root
    (:math:`alt(parent(n)) \\geq alt(n)` for any node :math:`n`) or decreasing. If :attr:`increasing_altitudes`
    is ``"auto"``, the monotony is deduced by comparing the altitude of the root with the smallest altitude among the
    internal nodes:

    .. math::

        increasing = \\left(alt(root) > \\min_{n \\textrm{ internal node}} alt(n)\\right)

    The altitudes of the leaves are not used. If all the internal nodes have the same altitude, the altitudes are
    considered decreasing.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(N)` time with ``"auto"``, where :math:`N` is the number of nodes in the
    tree, and :math:`\\mathcal{O}(1)` time otherwise.

    :Example:
    =======

    On the tree ``hg.Tree((5, 5, 6, 6, 8, 7, 7, 8, 8))``:

    .. code-block:: python

        # increasing_altitudes="auto"
        # altitudes (0, 0, 0, 0, 0, 1, 2, 4, 5)     -> True  (root 5 > minimum 1)
        # altitudes (0, 0, 0, 0, 0, -1, -2, -4, -5) -> False (root -5 is the minimum of the internal nodes)
        # altitudes (0, 0, 0, 0, 0, 3, 3, 3, 3)     -> False (constant altitudes)

    :param tree: input tree
    :param altitudes: node altitudes of the input tree (Numpy array of shape (tree.num_vertices(),))
    :param increasing_altitudes: ``True`` or ``"increasing"`` if the altitudes are increasing, ``False`` or
        ``"decreasing"`` if they are decreasing, and ``"auto"`` to deduce it from :attr:`altitudes`
    :return: ``True`` if the altitudes are increasing, ``False`` otherwise
    :raises ValueError: if :attr:`increasing_altitudes` is not one of the accepted values
    """
    if isinstance(increasing_altitudes, bool):
        return increasing_altitudes

    if increasing_altitudes == "auto":
        alt_root = altitudes[tree.root()]
        alt_min = np.min(altitudes[tree.num_leaves():])
        return bool(alt_root > alt_min)
    elif increasing_altitudes == "increasing":
        return True
    elif increasing_altitudes == "decreasing":
        return False
    else:
        raise ValueError("Unknown mode '" + str(increasing_altitudes) + "' valid values are 'auto', True, False, "
                                                                        "'increasing', and 'decreasing'.")


def attribute_extinction_value(tree, altitudes, attribute, increasing_altitudes="auto"):
    """
    Compute the extinction value of each node of the given tree for the given attribute.

    The extinction value of a node :math:`n` of the input tree :math:`T` with increasing altitudes :math:`alt`
    for the increasing attribute :math:`att` is equal to the threshold :math:`k` such that the node :math:`n`
    is still in a minimum of :math:`T` when all nodes having an attribute value smaller than :math:`k` are removed.

    Formally, let :math:`\\{M_i\\}` be the set of minima of the hierarchy :math:`T` with altitudes :math:`alt`.
    Let :math:`\\prec` be a total ordering of :math:`\\{M_i\\}` such that :math:`M_i \\prec M_j \\Rightarrow alt(M_i) \\leq alt(M_j)`.
    Let :math:`r(M_i)` be the smallest node of :math:`T` containing :math:`M_i` and another minimum :math:`M_j` such
    that :math:`M_j \\prec M_i`, and let :math:`c(M_i)` be the child of :math:`r(M_i)` containing :math:`M_i`.
    The extinction value of :math:`M_i` is then defined as :math:`att(c(M_i))`
    (for the deepest minimum, which has no such node, it is the attribute of the root).

    Extinction values of minima are then extended to other nodes in the tree with the following rules:

        - the extinction value of a non-leaf node :math:`n` which is not a minimum is defined as the largest
          extinction values among all the minima contained in :math:`n`
          (and 0 if :math:`n` does not contain any minima); and
        - the extinction value of a leaf node :math:`n` belonging to a minima :math:`M_i` is equal to the extinction
          value of :math:`M_i`. If :math:`n` does not belong to any minima its extinction value is 0.

    The function can also handle decreasing altitudes, in which case *minima* should be replaced by *maxima*
    in the description above. Possible values of :attr:`increasing_altitude` are:

        - ``'auto'``: the function will automatically determine if :attr:`altitudes` are increasing or decreasing (this has
          small computational cost but does not impact the runtime complexity).
        - ``True`` or ``'increasing'``: this means that altitudes are increasing from the leaves to the root
          (ie. for any node :math:`n`, :math:`altitudes(n) \leq altitudes(parent(n))`.
        - ``False`` or ``'decreasing'``: this means that altitudes are decreasing from the leaves to the root
          (ie. for any node :math:`n`, :math:`altitude(n) \geq altitude(parent(n))`.


    :Complexity:

    The computation takes :math:`\\mathcal{O}(N)` time, where :math:`N` is the number of nodes in the tree
    (plus :math:`\\mathcal{O}(N)` for the automatic detection of the monotony of the altitudes).

    :Example:
    =======

    Consider the following binary tree with 5 leaves:

    .. code-block:: text

        3         +-----8-----+
                  |           |
        2     +---7---+       |
              |       |       |
        1   +-5-+   +-6-+     |
            |   |   |   |     |
        0   0   1   2   3     4

    .. code-block:: python

        tree = hg.Tree((5, 5, 6, 6, 8, 7, 7, 8, 8))
        altitudes = np.asarray((0, 0, 0, 0, 0, 1, 2, 4, 5))
        area = hg.attribute_area(tree)
        hg.attribute_extinction_value(tree, altitudes, area)
        # array([5., 5., 2., 2., 0., 5., 2., 5., 5.])

    The nodes 5 (altitude 1) and 6 (altitude 2) are the two minima of the hierarchy and both have an area of 2.
    The node 5 is the deepest one: it never disappears, and its extinction value is the attribute of the root (5).
    The node 6 is merged into a deeper minimum when the area threshold exceeds its own area: its extinction value is 2.
    The leaves 0 and 1 (resp. 2 and 3) take the value of the minimum 5 (resp. 6); the leaf 4 does not belong to any
    minimum: its value is 0. The nodes 7 and 8 contain both minima: they take the largest of the two values.

    :param tree: input tree
    :param altitudes: node altitudes of the input tree (Numpy array of shape (tree.num_vertices(),))
    :param attribute: node attribute of the input tree, increasing from the leaves to the root (Numpy array of shape (tree.num_vertices(),))
    :param increasing_altitudes: possible values 'auto', True, False, 'increasing', and 'decreasing'
    :return: a 1d array like :attr:`attribute` representing the extinction value of each node of the tree
    """
    inc = __process_param_increasing_altitudes(tree, altitudes, increasing_altitudes)

    altitudes, attribute = hg.cast_to_common_type(altitudes, attribute)

    res = hg.cpp._attribute_extinction_value(tree, altitudes, attribute, inc)

    return res


@_legacy_cache_keywords
def attribute_height(tree, altitudes, increasing_altitudes="auto", reference_altitude="parent"):
    """
    Compute the height of each node of the given tree.

    In a tree :math:`T`, given that the altitudes of the nodes vary monotonically from the leaves to the root,
    the height :math:`h(n)` of a node :math:`n` of :math:`T` is the difference between the altitude of a reference
    node (either the parent of :math:`n` or :math:`n` itself, see :attr:`reference_altitude`) and the altitude of the
    deepest non-leaf node of the subtree of :math:`T` rooted in :math:`n`:

    .. math::

        h(n) = \\Big| alt(ref(n)) - \\min_{m \\in subtree(n) \\setminus leaves(n)} alt(m) \\Big|

    for increasing altitudes (the minimum is replaced by a maximum for decreasing altitudes), with
    :math:`ref(n) = parent(n)` by default. The height of a leaf is 0 and the parent of the root is the root.

    Possible values of :attr:`increasing_altitude` are:

        - ``'auto'``: the function will automatically determine if :attr:`altitudes` are increasing or decreasing (this has
          small computational cost but does not impact the runtime complexity).
        - ``True`` or ``'increasing'``: this means that altitudes are increasing from the leaves to the root
          (ie. for any node :math:`n`, :math:`altitudes(n) \leq altitudes(parent(n))`.
        - ``False`` or ``'decreasing'``: this means that altitudes are decreasing from the leaves to the root
          (ie. for any node :math:`n`, :math:`altitude(n) \geq altitude(parent(n))`.

    Possible values of :attr:`reference_altitude` are:

        - ``'parent'``: the reference altitude is the altitude of the parent of the current node.
        - ``'current'``: the reference altitude is the altitude of the current node.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(N)` time, where :math:`N` is the number of nodes in the tree
    (plus :math:`\\mathcal{O}(N)` for the automatic detection of the monotony of the altitudes).

    :Example:
    =======

    Consider the following binary tree with 5 leaves:

    .. code-block:: text

        3         +-----8-----+
                  |           |
        2     +---7---+       |
              |       |       |
        1   +-5-+   +-6-+     |
            |   |   |   |     |
        0   0   1   2   3     4

    .. code-block:: python

        tree = hg.Tree((5, 5, 6, 6, 8, 7, 7, 8, 8))
        altitudes = np.asarray((0, 0, 0, 0, 0, 1, 2, 4, 5))
        hg.attribute_height(tree, altitudes)
        # array([0, 0, 0, 0, 0, 3, 2, 4, 4])

    The deepest non-leaf node of the subtree rooted in the node 5 is the node 5 itself (altitude 1) and its parent,
    the node 7, has an altitude of 4: the height of the node 5 is :math:`4 - 1 = 3`. Similarly, the height of the node 6
    is :math:`4 - 2 = 2`. The deepest non-leaf node of the node 7 is the node 5 (altitude 1) and the altitude of its
    parent, the root, is 5: its height is :math:`5 - 1 = 4`. The root is its own parent and the deepest node of its
    subtree is the node 5: its height is also :math:`5 - 1 = 4`.

    :param tree: input tree
    :param altitudes: node altitudes of the input tree, monotonic from the leaves to the root (Numpy array of shape (tree.num_vertices(),))
    :param increasing_altitudes: possible values 'auto', True, False, 'increasing', and 'decreasing'
    :param reference_altitude: possible values 'parent' and 'current'
    :return: a 1d array like :attr:`altitudes` representing the height of each node of the tree
    """
    inc = __process_param_increasing_altitudes(tree, altitudes, increasing_altitudes)

    if reference_altitude == "current":
        if inc:
            res = hg.accumulate_sequential(tree,
                                           altitudes[:tree.num_leaves()],
                                           hg.Accumulators.max)\
                  - altitudes
        else:
            res = altitudes -\
                  hg.accumulate_sequential(tree,
                                           altitudes[:tree.num_leaves()],
                                           hg.Accumulators.min)
    else:
        res = hg.cpp._attribute_height(tree, altitudes, inc)

    return res


@_legacy_cache_keywords
def attribute_dynamics(tree, altitudes, increasing_altitudes="auto"):
    """
    Compute the dynamics of each node of the given tree.

    Given a node :math:`n` of the tree :math:`T`, the dynamics of :math:`n` is the difference between
    the altitude of the deepest minima of the subtree rooted in :math:`n` and the altitude
    of the closest ancestor of :math:`n` that has a deeper minima in its subtree. If no such
    ancestor exists then, the dynamics of :math:`n` is equal to the difference between the
    altitude of the highest node of the tree (the root) and the altitude of the deepest minima.

    The dynamics is the *extinction values* (:func:`~higra.attribute_extinction_value`) for the attribute *height*
    (:func:`~higra.attribute_height`).

    Possible values of :attr:`increasing_altitude` are:

        - ``'auto'``: the function will automatically determine if :attr:`altitudes` are increasing or decreasing (this has
          small computational cost but does not impact the runtime complexity).
        - ``True`` or ``'increasing'``: this means that altitudes are increasing from the leaves to the root
          (ie. for any node :math:`n`, :math:`altitudes(n) \leq altitudes(parent(n))`.
          (ie. for any node :math:`n`, :math:`altitude(n) \\geq altitude(parent(n))`.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(N)` time, where :math:`N` is the number of nodes in the tree
    (computation of :func:`~higra.attribute_height` followed by the computation of
    :func:`~higra.attribute_extinction_value`).

    :Example:
    =======

    Consider the following binary tree with 5 leaves:

    .. code-block:: text

        3         +-----8-----+
                  |           |
        2     +---7---+       |
              |       |       |
        1   +-5-+   +-6-+     |
            |   |   |   |     |
        0   0   1   2   3     4

    .. code-block:: python

        tree = hg.Tree((5, 5, 6, 6, 8, 7, 7, 8, 8))
        altitudes = np.asarray((0, 0, 0, 0, 0, 1, 2, 4, 5))
        hg.attribute_dynamics(tree, altitudes)
        # array([4., 4., 2., 2., 0., 4., 2., 4., 4.])

    The hierarchy has two minima: the nodes 5 (altitude 1) and 6 (altitude 2). The node 6 is merged with a deeper
    minimum in the node 7 (altitude 4): its dynamics is :math:`4 - 2 = 2`. The node 5 is the deepest minimum: its
    dynamics is the difference between the altitude of the root and its own altitude, :math:`5 - 1 = 4`.
    The leaves 0 and 1 (resp. 2 and 3) take the value of the minimum 5 (resp. 6), the leaf 4 does not belong to any
    minimum, and the nodes 7 and 8, which contain both minima, take the largest value.

    :param tree: input tree
    :param altitudes: node altitudes of the input tree, monotonic from the leaves to the root (Numpy array of shape (tree.num_vertices(),))
    :param increasing_altitudes: possible values 'auto', True, False, 'increasing', and 'decreasing'
    :return: a 1d array representing the dynamics of each node of the tree
    """

    inc = __process_param_increasing_altitudes(tree, altitudes, increasing_altitudes)

    height = hg.attribute_height(tree, altitudes, inc)

    return hg.attribute_extinction_value(tree, altitudes, height, inc)


@_legacy_cache_keywords
def attribute_child_number(tree):
    """
    Compute the rank of each node in the ordered list of children of its parent.

    Given a node :math:`n` whose parent is :math:`p`, the attribute value of :math:`n` is the rank of :math:`n`
    in the list of children of :math:`p` (starting from 0):

    .. math::

        cn(n) = i \\iff n = child(p, i)

    where :math:`child(p, i)` is the :math:`i`-th child of :math:`p`. The root of the tree has no parent and takes the
    value -1.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(N)` time, where :math:`N` is the number of nodes in the tree.

    :Example:
    =======

    Consider the following binary tree with 5 leaves:

    .. code-block:: text

        3         +-----8-----+
                  |           |
        2     +---7---+       |
              |       |       |
        1   +-5-+   +-6-+     |
            |   |   |   |     |
        0   0   1   2   3     4

    .. code-block:: python

        tree = hg.Tree((5, 5, 6, 6, 8, 7, 7, 8, 8))
        hg.attribute_child_number(tree)
        # array([ 0,  1,  0,  1,  0,  0,  1,  1, -1])

    The leaves 0 and 1 are the first and the second child of the node 5. The leaf 4 is the first child of the root and the
    node 7 is its second child. The root has no parent: its value is -1.

    :param tree: input tree
    :return: a 1d array of integers representing the rank of each node of the tree among the children of its parent
    """

    res = hg.cpp._attribute_child_number(tree)

    return res


def attribute_children_pair_sum_product(tree, node_weights):
    """
    Compute, for each node, the sum of the products of the weights of every pair of its children.

    Given a tree :math:`T` with node weights :math:`w`, the children pair sum product for a node :math:`n` sums, for
    every pair :math:`(c_i, c_j)` of distinct children of :math:`n`, the product of the node weights of :math:`c_i` and
    :math:`c_j`. Formally:

    .. math::

        res(n) = \\sum_{i=0}^{numc(n) - 1} \\sum_{j=0}^{i - 1} w(child(i, n)) \\times w(child(j, n))

    where :math:`numc(n)` is the number of children of :math:`n` and :math:`child(i, n)` is the :math:`i`-th child
    of the node :math:`n`. The value of a leaf, which has no child, is 0.

    If the node weights are vectors, the computation is done independently for each component. The result is thus an
    array with the same shape as :attr:`node_weights`.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(N \\times C^2 \\times d)` time, where :math:`N` is the number of nodes in
    the tree, :math:`C` is the maximal number of children of a node, and :math:`d` is the dimension of the node weights
    (:math:`d = 1` for scalar weights).

    :Example:
    =======

    Consider the following binary tree with 5 leaves:

    .. code-block:: text

        3         +-----8-----+
                  |           |
        2     +---7---+       |
              |       |       |
        1   +-5-+   +-6-+     |
            |   |   |   |     |
        0   0   1   2   3     4

    .. code-block:: python

        tree = hg.Tree((5, 5, 6, 6, 8, 7, 7, 8, 8))
        node_weights = np.asarray((1., 2, 3, 4, 5, 3, 7, 10, 15))
        hg.attribute_children_pair_sum_product(tree, node_weights)
        # array([ 0.,  0.,  0.,  0.,  0.,  2., 12., 21., 50.])

    Each non-leaf node of this binary tree has exactly one pair of children: the result for the node 5 is
    :math:`1 \\times 2 = 2`, for the node 6 is :math:`3 \\times 4 = 12`, for the node 7 is
    :math:`3 \\times 7 = 21` (weights of the nodes 5 and 6) and for the root is :math:`5 \\times 10 = 50` (weights of the
    leaf 4 and of the node 7).

    :param tree: input tree
    :param node_weights: node weights of the input tree (Numpy array whose first dimension is ``tree.num_vertices()``)
    :return: an array with the same shape as :attr:`node_weights`
    """

    res = hg.cpp._attribute_children_pair_sum_product(tree, node_weights)

    return res


def attribute_tree_sampling_probability(tree, leaf_graph, leaf_graph_edge_weights, model='edge'):
    """
    Compute, for each node of the given tree, the probability that it is the smallest cluster containing a pair of
    vertices sampled in the graph :math:`G=(V, E)` with edge weights :math:`w`.

    This method is defined in [1]_.

    The probability :math:`P(\\{a,b\\})` of a pair of vertices :math:`\\{a,b\\}` is :math:`w(\\{a,b\\}) / Z`
    with :math:`Z=\\sum_{e\\in E}w(e)` if :math:`\\{a,b\\}` is an edge of :math:`G`, and 0 otherwise.
    The probability :math:`P(a)` of a vertex :math:`a` is then :math:`\\sum_{b\\in V}P(\\{a, b\\})`.

    Two sampling strategies are proposed for sampling pairs of vertices to compute the probability of a node of the tree:

    - *edge*: the probability of sampling the pair :math:`\\{a, b\\}` is given by :math:`P(\\{a, b\\})`; and
    - *null*: the probability of sampling the pair :math:`\\{a, b\\}` is given by the product of the probabilities
      of :math:`a` and :math:`b`: :math:`P(a) \\times P(b)`.

    The probability of a node :math:`n` is the probability that a sampled pair :math:`\\{a, b\\}` has :math:`n` as lowest common ancestor:

    .. math::

        p_{edge}(n) = \\sum_{e = \\{a, b\\} \\in E,\\ lca(a, b) = n} P(e)
        \\qquad\\qquad
        p_{null}(n) = \\sum_{\\{a, b\\} \\subset V,\\ lca(a, b) = n} P(a) \\times P(b)

    where :math:`p_{null}(n)` is computed as the sum, over every pair :math:`\\{c_i, c_j\\}` of children of :math:`n`, of
    :math:`P(c_i) \\times P(c_j)`, with :math:`P(c)` the sum of the probabilities of the vertices of :math:`c`
    (see :func:`~higra.attribute_children_pair_sum_product`). The probability of a leaf is 0.

    Assuming that the edge weights on the leaf graph of a hierarchy represents similarities:

    .. epigraph::

        *We expect these distributions to differ significantly if the tree indeed represents the hierarchical structure of the graph.
        Specifically, we expect [the edge distribution] to be mostly concentrated on deep nodes of the tree
        (far from the root), as two nodes* :math:`u`, :math:`v` *connected with high weight* :math:`w(\\{u, v\\})` *in the graph
        typically  belong to a small cluster, representative of the clustering structure of the graph; on the contrary,
        we expect [the null distribution] to be concentrated over shallow nodes (close to the root) as two nodes*
        :math:`w(\{u, v\})` *sampled independently at random typically belong to large clusters, less representative of the
        clustering structure of the graph*. [1]_


    .. [1] Charpentier, B. & Bonald, T. (2019).  `"Tree Sampling Divergence: An Information-Theoretic Metric for \
           Hierarchical Graph Clustering." <https://hal.telecom-paristech.fr/hal-02144394/document>`_ Proceedings of IJCAI.

    :Complexity:

    The tree sampling divergence runtime complexity depends of the sampling model:

     - *edge*: :math:`\mathcal{O}(N\log(N) + M)` with :math:`N` the number of  nodes in the tree and :math:`M` the number of edges in the leaf graph.
     - *null*: :math:`\mathcal{O}(N\\times C^2)` with :math:`N` the number of nodes in the tree  and :math:`C` the maximal number of children of a node in the tree.

    :Example:
    =======

    Consider the following 3 x 3 grid graph, whose edges are labelled with their weights (the similarities):

    .. code-block:: text

        (0)-- 0 --(1)-- 2 --(2)
        |         |         |
        6         6         0
        |         |         |
        (3)-- 0 --(4)-- 4 --(5)
        |         |         |
        5         5         3
        |         |         |
        (6)-- 0 --(7)-- 1 --(8)

    and the binary partition tree computed on it with :func:`~higra.bpt_canonical`:

    .. code-block:: text

        4                 +-------16------+
                          |               |
        3         +-------15-----+        |
                  |              |        |
        2     +---14--+          |        |
              |       |          |        |
        1     |       |       +--13-+     |
              |       |       |     |     |
        0   +-9-+   +-10+   +-12+   |   +-11+
            +   +   +   +   +   +   +   +   +
            0   1   2   5   6   7   8   3   4

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0, 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1))

        tree, altitudes = hg.bpt_canonical(graph, edge_weights)
        hg.attribute_tree_sampling_probability(tree, graph, edge_weights, model='edge')
        # array([0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
        #        0.     , 0.     , 0.     , 0.     , 0.03125, 0.0625 , 0.09375, 0.8125 ])

        hg.attribute_tree_sampling_probability(tree, graph, edge_weights, model='null')
        # array([0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
        #        0.        , 0.046875  , 0.01367188, 0.16113281, 0.02929688, 0.04296875, 0.12304688, 0.33691406,
        #        0.96484375])

    With :math:`Z = 0 + 6 + 2 + 6 + 0 + 0 + 5 + 4 + 5 + 3 + 0 + 1 = 32`, the lowest common ancestor of each edge is given by
    :func:`~higra.attribute_lca_map`: the edge :math:`\\{7, 8\\}` (weight 1) has the node 13 as lowest common ancestor, the edge :math:`\\{1, 2\\}`
    (weight 2) the node 14, and the edge :math:`\\{5, 8\\}` (weight 3) the node 15. Their probabilities are respectively
    :math:`1/32`, :math:`2/32` and :math:`3/32`. The five remaining edges of non-zero weight (6, 6, 5, 4 and 5) have the root as lowest
    common ancestor: its probability is :math:`26/32 = 0.8125`. The edges of weight 0 do not contribute.
    With the null model, the probabilities of the vertices 0 and 1 are :math:`P(0) = 6/32` and :math:`P(1) = 8/32`
    (the sums of the weights of their incident edges divided by :math:`Z`), hence a probability of
    :math:`P(0) \\times P(1) = 48/1024 = 0.046875` for the node 9.

    :see:

    The :func:`~higra.tree_sampling_divergence` is a non supervised hierarchical cost function defined as the
    Kullback-Leibler divergence between the edge sampling model and the independent (null) sampling model.

    :param tree: input tree
    :param leaf_graph: graph defined on the leaves of the input tree
    :param leaf_graph_edge_weights: edge weights of the leaf graph (similarities), one value per edge (Numpy array of shape (leaf_graph.num_edges(),))
    :param model: defines the edge sampling strategy, either "edge" or "null"
    :return: a 1d array representing, for each node of the tree, the probability that it is the smallest cluster containing a pair of vertices sampled with the chosen model
    """
    if model not in ("edge", "null"):
        raise ValueError("Parameter 'model' must be either 'edge' or 'null'.")

    if model == 'edge':
        lca_map = hg.attribute_lca_map(tree, leaf_graph=leaf_graph)
        leaf_graph_edge_weights = leaf_graph_edge_weights / np.sum(leaf_graph_edge_weights)
        return hg.accumulate_at(lca_map, leaf_graph_edge_weights, hg.Accumulators.sum)
    else:  # model = 'null'
        leaf_graph_vertex_weights = hg.accumulate_graph_edges(leaf_graph, leaf_graph_edge_weights, hg.Accumulators.sum)
        leaf_graph_vertex_weights = leaf_graph_vertex_weights / np.sum(leaf_graph_edge_weights)
        tree_node_weights = hg.accumulate_sequential(tree, leaf_graph_vertex_weights, hg.Accumulators.sum)
        return hg.attribute_children_pair_sum_product(tree, tree_node_weights)


@_legacy_cache_keywords
def attribute_topological_height(tree):
    """
    Compute the topological height of each node of the given tree.

    The topological height :math:`h(n)` of a node :math:`n` is the number of edges on the longest path from :math:`n`
    to a leaf of the subtree rooted in :math:`n`:

    .. math::

        h(n) = \\begin{cases}
            0 & \\textrm{if } n \\textrm{ is a leaf}\\\\
            1 + \\max_{c \\in children(n)} h(c) & \\textrm{otherwise}
        \\end{cases}

    Unlike :func:`~higra.attribute_depth`, which is measured from the root, the topological height is measured from the
    leaves. The height of the root is the length of the longest path from the root to a leaf.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(N)` time, where :math:`N` is the number of nodes in the tree.

    :Example:
    =======

    Consider the following binary tree with 5 leaves:

    .. code-block:: text

        3         +-----8-----+
                  |           |
        2     +---7---+       |
              |       |       |
        1   +-5-+   +-6-+     |
            |   |   |   |     |
        0   0   1   2   3     4

    .. code-block:: python

        tree = hg.Tree((5, 5, 6, 6, 8, 7, 7, 8, 8))
        hg.attribute_topological_height(tree)
        # array([0, 0, 0, 0, 0, 1, 1, 2, 3])

    The nodes 5 and 6 are one edge above their leaves, the node 7 is two edges above the leaves 0 to 3, and the root is
    three edges above them (through the nodes 7 and 5 or 6): the leaf 4 only gives a path of length 1.

    :param tree: input tree
    :return: a 1d array of integers representing the topological height of each node of the tree
    """

    res = hg.accumulate_and_add_sequential(tree,
                                           np.ones(tree.num_vertices(), dtype=np.int64),
                                           np.zeros(tree.num_leaves(), dtype=np.int64),
                                           hg.Accumulators.max)

    return res


@_legacy_cache_keywords
@hg.argument_helper(hg.CptHierarchy)
def attribute_moment_of_inertia(tree, leaf_graph):
    """
    Compute the moment of inertia (first Hu moment) of each node of the given tree.

    The moment of inertia is a translation, scale and rotation invariant characterization of the shape of the nodes.
    This function works only if :attr:`leaf_graph` is a 2D grid graph (e.g. the graph of an image): otherwise a
    ``ValueError`` is raised.

    Given a node :math:`X` of :attr:`tree`, the raw moments :math:`M_{ij}` are defined as:

    .. math::

        M_{ij} = \\sum_{(x, y) \\in X} x^i y^j

    where :math:`(x,y)` are the coordinates of every vertex in :math:`X` (the leaves of the subtree rooted in the node).
    Then, the centroid :math:`(\\overline{x},\\overline{y})` of :math:`X` is given by

    .. math::

         \\overline{x} = \\frac{M_{10}}{M_{00}} \\qquad\\textrm{and}\\qquad \\overline{y} = \\frac{M_{01}}{M_{00}}

    Some central moments of :math:`X` are then:

    - :math:`\\mu_{00} = M_{00}`
    - :math:`\\mu_{20} = M_{20} - \\overline{x} \\times M_{10}`
    - :math:`\\mu_{02} = M_{02} - \\overline{y} \\times M_{01}`

    The moment of inertia :math:`I_1` of :math:`X` is finally defined as

    .. math::

        I_1 = \\eta_{20} + \\eta_{02}

    where :math:`\\eta_{ij}` are given by:

    .. math::

        \\eta_{ij} = \\frac{\\mu_{ij}}{\\mu_{00}^{1+\\frac{i+j}{2}}}

    The moment of inertia of a leaf (a single pixel) is 0.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(N + n)` time, where :math:`N` is the number of nodes in the tree and
    :math:`n` is the number of vertices of the leaf graph (computation of the coordinates of the vertices
    followed by one accumulation of the moments over the tree).

    :Example:
    =======

    Consider the following 3 x 3 grid graph, whose edges are labelled with their weights:

    .. code-block:: text

        (0)-- 0 --(1)-- 2 --(2)
        |         |         |
        6         6         0
        |         |         |
        (3)-- 0 --(4)-- 4 --(5)
        |         |         |
        5         5         3
        |         |         |
        (6)-- 0 --(7)-- 1 --(8)

    and the binary partition tree computed on it with :func:`~higra.bpt_canonical`:

    .. code-block:: text

        4                 +-------16------+
                          |               |
        3         +-------15-----+        |
                  |              |        |
        2     +---14--+          |        |
              |       |          |        |
        1     |       |       +--13-+     |
              |       |       |     |     |
        0   +-9-+   +-10+   +-12+   |   +-11+
            +   +   +   +   +   +   +   +   +
            0   1   2   5   6   7   8   3   4

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0., 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1))

        tree, altitudes = hg.bpt_canonical(graph, edge_weights)
        result = hg.attribute_moment_of_inertia(tree)
        # result[0]  == 0.
        # result[9]  == 0.125
        # result[13] == 0.22222222
        # result[16] == 0.14814815

    The node 9 is made of the two adjacent pixels :math:`(0, 0)` and :math:`(0, 1)`: :math:`M_{00} = 2` and
    :math:`\\mu_{20} + \\mu_{02} = 0 + 0.5`, hence :math:`I_1 = 0.5 / 2^2 = 0.125`.
    The node 13 is the bottom row, made of the pixels :math:`(2, 0)`, :math:`(2, 1)` and :math:`(2, 2)`: :math:`\\mu_{20} = 0`,
    :math:`\\mu_{02} = 2` and :math:`M_{00} = 3`, hence :math:`I_1 = 2 / 3^2 = 0.2222`.
    The root is the whole image: :math:`\\mu_{20} = \\mu_{02} = 6` and :math:`M_{00} = 9`, hence
    :math:`I_1 = 12 / 9^2 = 0.1481`.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param leaf_graph: graph on the leaves of the input tree, which must be a 2D grid graph (deduced from :class:`~higra.CptHierarchy` on `tree`)
    :return: a 1d array of floating point values representing the moment of inertia of each node of the tree
    """

    if (not hg.CptGridGraph.validate(leaf_graph)) or (len(hg.CptGridGraph.get_shape(leaf_graph)) != 2):
        raise ValueError("Parameter 'leaf_graph' must be a 2D grid graph.")

    coordinates = hg.attribute_vertex_coordinates(leaf_graph)
    coordinates = np.reshape(coordinates, (coordinates.shape[0] * coordinates.shape[1], coordinates.shape[2]))

    M_00_leaves = np.ones((tree.num_leaves())).astype(dtype=np.float64)
    x_leaves = coordinates[:, 0].astype(dtype=np.float64)
    y_leaves = coordinates[:, 1].astype(dtype=np.float64)

    M_10_leaves = x_leaves * M_00_leaves
    M_01_leaves = y_leaves * M_00_leaves
    M_20_leaves = (x_leaves ** 2) * M_00_leaves
    M_02_leaves = (y_leaves ** 2) * M_00_leaves

    M_00 = hg.accumulate_sequential(tree, M_00_leaves, hg.Accumulators.sum)
    M_01 = hg.accumulate_sequential(tree, M_01_leaves, hg.Accumulators.sum)
    M_10 = hg.accumulate_sequential(tree, M_10_leaves, hg.Accumulators.sum)
    M_02 = hg.accumulate_sequential(tree, M_02_leaves, hg.Accumulators.sum)
    M_20 = hg.accumulate_sequential(tree, M_20_leaves, hg.Accumulators.sum)

    _x = M_10 / M_00
    _y = M_01 / M_00
    miu_20 = M_20 - _x * M_10
    miu_02 = M_02 - _y * M_01
    I_1 = (miu_20 + miu_02) / (M_00 ** 2)

    return I_1

@_legacy_cache_keywords
@hg.argument_helper(hg.CptHierarchy)
def attribute_node_bounding_box(tree, leaf_graph):
    """
    Compute the axis-aligned bounding box of every node of a tree.

    Let ``G`` be the leaf graph of ``tree``. The graph is assumed to be a grid
    graph of spatial dimension :math:`d`, with:

    .. math::

        d \\in \\mathbb{N}^*

    Each vertex ``v`` of ``G`` is associated with a coordinate vector:

    .. math::

        p(v) =
        \\left(
            p_0(v), p_1(v), \\ldots, p_{d-1}(v)
        \\right)

    For a node ``n`` of the tree, let ``V(n)`` denote the set of leaf graph
    vertices contained in the subtree rooted at ``n``.

    For every spatial dimension ``k``, the minimum and maximum coordinates of
    ``n`` are computed independently:

    .. math::

        \\operatorname{min\\_coordinates}(n, k)
        =
        \\min_{v \\in V(n)} p_k(v)

    .. math::

        \\operatorname{max\\_coordinates}(n, k)
        =
        \\max_{v \\in V(n)} p_k(v)

    The axis-aligned bounding box of node ``n`` is therefore the Cartesian
    product of the intervals defined along every spatial dimension:

    .. math::

        B(n)
        =
        \\prod_{k=0}^{d-1}
        \\left[
            \\operatorname{min\\_coordinates}(n, k),
            \\operatorname{max\\_coordinates}(n, k)
        \\right]

    The result is a NumPy array with shape ``(N, d, 2)``, where:

    - ``N`` is the number of nodes in ``tree``;
    - ``d`` is the spatial dimension of ``leaf_graph``.

    The axes of the result are organized as follows:

    .. code-block:: text

        axis 0: tree node
        axis 1: spatial coordinate
        axis 2: minimum or maximum

    :Example:
    =======

    Consider the following 3 x 3 grid graph, whose edges are labelled with their weights:

    .. code-block:: text

        (0)-- 0 --(1)-- 2 --(2)
        |         |         |
        6         6         0
        |         |         |
        (3)-- 0 --(4)-- 4 --(5)
        |         |         |
        5         5         3
        |         |         |
        (6)-- 0 --(7)-- 1 --(8)

    and the binary partition tree computed on it with :func:`~higra.bpt_canonical`:

    .. code-block:: python

        4                 +-------16------+
                          |               |
        3         +-------15-----+        |
                  |              |        |
        2     +---14--+          |        |
              |       |          |        |
        1     |       |       +--13-+     |
              |       |       |     |     |
        0   +-9-+   +-10+   +-12+   |   +-11+
            +   +   +   +   +   +   +   +   +
            0   1   2   5   6   7   8   3   4

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0, 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1))

        tree, _ = hg.bpt_canonical(graph, edge_weights)
        result = hg.attribute_node_bounding_box(tree)

    The bounding box of the leaf node 6 is:
    
        .. code-block:: python
    
            result[6] == [[2, 2],[0, 0]]    

    Its minimum and maximum corners are respectively:

    .. code-block:: python

        minimum_corner = result[6, :, 0]  # [2, 0]
        maximum_corner = result[6, :, 1]  # [2, 0]

    The bounding box of node 13 is:

    .. code-block:: python

        result[13] == [[2, 2],[0, 2]]

    This means that:

    .. math::

        x \\in [2, 2]
        \\qquad\\text{and}\\qquad
        y \\in [0, 2]

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param leaf_graph: leaf graph associated with ``tree`` (Concept :class:`~higra.CptGridGraph`).
        Its vertices must provide coordinates in a grid of dimension :math:`d`,
        with :math:`d \\in \\mathbb{N}^*`.
    :return: a NumPy array of shape ``(N, d, 2)`` containing the minimum and
        maximum coordinates of every tree node.
"""
    
    coordinates = hg.attribute_vertex_coordinates(leaf_graph)
    coordinates = np.reshape(coordinates, (coordinates.shape[0] * coordinates.shape[1], coordinates.shape[2]))

    min_coordinates = hg.accumulate_sequential(tree, coordinates, hg.Accumulators.min, leaf_graph)
    max_coordinates = hg.accumulate_sequential(tree, coordinates, hg.Accumulators.max, leaf_graph)

    min_max_coordinates = np.stack((min_coordinates, max_coordinates), axis=-1)

    return min_max_coordinates