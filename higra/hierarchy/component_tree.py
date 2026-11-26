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


def component_tree_min_tree(graph, vertex_weights):
    """
    Compute the min tree of the given vertex weighted graph.

    For a threshold :math:`k`, the lower level set :math:`L_k` of the vertex weights :math:`w` is the set of the
    vertices whose weight is smaller than or equal to :math:`k`:

    .. math::

        L_k = \\{v \\in V \\mid w(v) \\leq k\\}

    where :math:`V` is the vertex set of the graph. The min tree is the tree of the connected components of the lower
    level sets of :math:`w`, for all thresholds :math:`k`, ordered by inclusion. Each component appears only once,
    even if it is a connected component of several lower level sets: its altitude is the smallest threshold for which
    it is a connected component, that is the largest weight of its vertices:

    .. math::

        \\mathrm{alt}(C) = \\max_{v \\in C} w(v)

    The min tree was proposed in [#]_ and [#]_, and the algorithm used in this implementation was first described
    in [#]_.

    The resulting tree is made of:

    - the leaves, which are the vertices of the graph: the leaf :math:`v` has the altitude :math:`w(v)`, and its
      parent is the connected component of :math:`L_{w(v)}` containing :math:`v`, which has the same altitude as
      :math:`v`;
    - the non-leaf nodes, which are the connected components of the lower level sets. The altitudes are increasing
      from the leaves to the root (for any node :math:`n`, :math:`\\mathrm{alt}(n) \\leq \\mathrm{alt}(parent(n))`),
      and a non-leaf node never has the same altitude as its parent, except the root, which is its own parent. The
      root is the whole vertex set, with the largest vertex weight as altitude.

    The graph must have at least one vertex (an exception is raised otherwise) and must be connected (the result is
    undefined otherwise). The vertex weights must be scalars: :attr:`vertex_weights` is an array with one element per
    vertex or, if the graph is a grid graph (e.g. an image), an array of the shape of the grid.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(n\\log(n) + m\\,\\alpha(n))` time, where :math:`n` is the number of
    vertices and :math:`m` the number of edges of the graph, and :math:`\\alpha` is the inverse Ackermann function
    (sorting of the vertices by weight followed by a union-find over the edges).

    :Example:
    =======

    Consider the following 1 x 6 image, whose pixels (the vertices of a 4-adjacency graph) are labelled with their
    weights:

    .. code-block:: text

        pixel   0  1  2  3  4  5
        weight  1  3  2  3  0  2

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((1, 6))
        vertex_weights = np.asarray(((1, 3, 2, 3, 0, 2),))
        tree, altitudes = hg.component_tree_min_tree(graph, vertex_weights)
        tree.parents()
        # array([ 7, 10,  9, 10,  6,  8,  8, 10, 10, 10, 10])
        altitudes
        # array([1, 3, 2, 3, 0, 2, 0, 1, 2, 2, 3])

    The leaves 0 to 5 are the pixels and the nodes 6 to 10 are the connected components of the lower level sets
    (altitudes on the left):

    .. code-block:: text

        3           +-----------10-------------+
                    |     |     |     |        |
        2           |     |     9     |     +--8--+
                    |     |     |     |     |     |
        1           7     |     |     |     |     |
                    |     |     |     |     |     |
        0           |     |     |     |     6     |
                    |     |     |     |     |     |
                    0     1     2     3     4     5

    The node 6 is the component :math:`\\{4\\}` of :math:`L_0`. It is also a component of :math:`L_1`, and thus
    appears only once. The node 8 is the component :math:`\\{4, 5\\}` of :math:`L_2`, whose largest weight is 2.
    The pixels 1 and 3 have the largest weight: their parent is the root, the whole image, of altitude 3.

    .. [#] Ph. Salembier, A. Oliveras, and L. Garrido, "Anti-extensive connected operators for image \
           and sequence processing," IEEE Trans. Image Process., vol. 7, no. 4, pp. 555-570, Apr. 1998.
    .. [#] Ro. Jones, "Connected filtering and segmentation using component trees," Comput. Vis. \
           Image Understand., vol. 75, no. 3, pp. 215-228, Sep. 1999.
    .. [#] Ch. Berger, T. Geraud, R. Levillain, N. Widynski, A. Baillard, and E. Bertin, "Effective \
           Component Tree Computation with Application to Pattern Recognition in Astronomical Imaging," \
           IEEE ICIP 2007.

    :param graph: input graph, connected and with at least one vertex
    :param vertex_weights: weight of the vertices of the input graph (scalar values)
    :return: a tree (Concept :class:`~higra.CptHierarchy`, whose leaf graph is :attr:`graph`) and its node altitudes
        (a 1d array with one element per node of the tree, with the same dtype as :attr:`vertex_weights`)
    """
    vertex_weights = hg.linearize_vertex_weights(vertex_weights, graph)

    tree, altitudes = hg.cpp._component_tree_min_tree(graph, vertex_weights)

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def component_tree_max_tree(graph, vertex_weights):
    """
    Compute the max tree of the given vertex weighted graph.

    For a threshold :math:`k`, the upper level set :math:`U_k` of the vertex weights :math:`w` is the set of the
    vertices whose weight is greater than or equal to :math:`k`:

    .. math::

        U_k = \\{v \\in V \\mid w(v) \\geq k\\}

    where :math:`V` is the vertex set of the graph. The max tree is the tree of the connected components of the upper
    level sets of :math:`w`, for all thresholds :math:`k`, ordered by inclusion. Each component appears only once,
    even if it is a connected component of several upper level sets: its altitude is the largest threshold for which
    it is a connected component, that is the smallest weight of its vertices:

    .. math::

        \\mathrm{alt}(C) = \\min_{v \\in C} w(v)

    The max tree was proposed in [#]_ and [#]_, and the algorithm used in this implementation was first described
    in [#]_.

    The resulting tree is made of:

    - the leaves, which are the vertices of the graph: the leaf :math:`v` has the altitude :math:`w(v)`, and its
      parent is the connected component of :math:`U_{w(v)}` containing :math:`v`, which has the same altitude as
      :math:`v`;
    - the non-leaf nodes, which are the connected components of the upper level sets. The altitudes are decreasing
      from the leaves to the root (for any node :math:`n`, :math:`\\mathrm{alt}(n) \\geq \\mathrm{alt}(parent(n))`),
      and a non-leaf node never has the same altitude as its parent, except the root, which is its own parent. The
      root is the whole vertex set, with the smallest vertex weight as altitude.

    The graph must have at least one vertex (an exception is raised otherwise) and must be connected (the result is
    undefined otherwise). The vertex weights must be scalars: :attr:`vertex_weights` is an array with one element per
    vertex or, if the graph is a grid graph (e.g. an image), an array of the shape of the grid.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(n\\log(n) + m\\,\\alpha(n))` time, where :math:`n` is the number of
    vertices and :math:`m` the number of edges of the graph, and :math:`\\alpha` is the inverse Ackermann function
    (sorting of the vertices by weight followed by a union-find over the edges).

    :Example:
    =======

    Consider the following 1 x 6 image, whose pixels (the vertices of a 4-adjacency graph) are labelled with their
    weights:

    .. code-block:: text

        pixel   0  1  2  3  4  5
        weight  1  3  2  3  0  2

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((1, 6))
        vertex_weights = np.asarray(((1, 3, 2, 3, 0, 2),))
        tree, altitudes = hg.component_tree_max_tree(graph, vertex_weights)
        tree.parents()
        # array([10,  7,  9,  6, 11,  8,  9,  9, 11, 10, 11, 11])
        altitudes
        # array([1, 3, 2, 3, 0, 2, 3, 3, 2, 2, 1, 0])

    The leaves 0 to 5 are the pixels and the nodes 6 to 11 are the connected components of the upper level sets
    (altitudes on the left):

    .. code-block:: text

        0                 +----------11-----------+
                          |                 |     |
        1           +----10-----+           |     |
                    |           |           |     |
        2           |     +-----9-----+     |     8
                    |     |     |     |     |     |
        3           |     7     |     6     |     |
                    |     |     |     |     |     |
                    0     1     2     3     4     5

    The nodes 7 and 6 are the two components :math:`\\{1\\}` and :math:`\\{3\\}` of :math:`U_3`: the pixel 2, of
    weight 2, separates them. They are merged in the node 9, the component :math:`\\{1, 2, 3\\}` of :math:`U_2`. The
    node 8 is the component :math:`\\{5\\}` of :math:`U_2`; it is also a component of :math:`U_1`, and thus appears
    only once. The pixel 4 has the smallest weight: its parent is the root, the whole image, of altitude 0.

    .. [#] Ph. Salembier, A. Oliveras, and L. Garrido, "Anti-extensive connected operators for image \
           and sequence processing," IEEE Trans. Image Process., vol. 7, no. 4, pp. 555-570, Apr. 1998.
    .. [#] Ro. Jones, "Connected filtering and segmentation using component trees," Comput. Vis. \
           Image Understand., vol. 75, no. 3, pp. 215-228, Sep. 1999.
    .. [#] Ch. Berger, T. Geraud, R. Levillain, N. Widynski, A. Baillard, and E. Bertin, "Effective \
           Component Tree Computation with Application to Pattern Recognition in Astronomical Imaging," \
           IEEE ICIP 2007.

    :param graph: input graph, connected and with at least one vertex
    :param vertex_weights: weight of the vertices of the input graph (scalar values)
    :return: a tree (Concept :class:`~higra.CptHierarchy`, whose leaf graph is :attr:`graph`) and its node altitudes
        (a 1d array with one element per node of the tree, with the same dtype as :attr:`vertex_weights`)
    """
    vertex_weights = hg.linearize_vertex_weights(vertex_weights, graph)

    tree, altitudes = hg.cpp._component_tree_max_tree(graph, vertex_weights)

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes
