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


def binary_partition_tree_complete_linkage(graph, edge_weights):
    """
    Binary partition tree with complete linkage distance.

    Given a graph :math:`G=(V, E)`, with initial edge weights :math:`w`,
    the distance :math:`d(X,Y)` between any two clusters :math:`X` and :math:`Y` is

    .. math::

        d(X,Y) = \max \{w(\{x,y\}) | x \in X, y \in Y, \{x,y\} \in E \}

    Regions are then iteratively merged following the above distance (closest first) until a single region remains

    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    res = hg.cpp._binary_partition_tree_complete_linkage(graph, edge_weights)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def binary_partition_tree_average_linkage(graph, edge_weights, edge_weight_weights=None):
    """
    Binary partition tree with average linkage distance.

    Given a graph :math:`G=(V, E)`, with initial edge weights :math:`w` with associated weights :math:`w_2`,
    the distance :math:`d(X,Y)` between any two clusters :math:`X` and :math:`Y` is

    .. math::

        d(X,Y) = \\frac{1}{Z} \sum_{x \in X, y \in Y, \{x,y\} \in E} w(\{x,y\}) \\times w_2(\{x,y\})

    with :math:`Z = \sum_{x \in X, y \in Y, \{x,y\} \in E} w_2({x,y})`.

    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :param edge_weight_weights: weighting of edge weights of the input graph (default to an array of ones)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    if edge_weight_weights is None:
        edge_weight_weights = np.ones_like(edge_weights)
    else:
        edge_weights, edge_weight_weights = hg.cast_to_common_type(edge_weights, edge_weight_weights)

    res = hg.cpp._binary_partition_tree_average_linkage(graph, edge_weights, edge_weight_weights)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def binary_partition_tree_exponential_linkage(graph, edge_weights, alpha, edge_weight_weights=None):
    """
    Binary partition tree with exponential linkage distance.

    Given a graph :math:`G=(V, E)`, with initial edge weights :math:`w` with associated weights :math:`w_2`,
    the distance :math:`d(X,Y)` between any two clusters :math:`X` and :math:`Y` is

    .. math::

         d(X,Y) = \\frac{1}{Z} \sum_{x \in X, y \in Y, \{x,y\} in E} w_2(\{x,y\}) \\times \exp(\\alpha * w(\{x,y\})) \\times w(\{x,y\})

    with :math:`Z = \sum_{x \in X, y \in Y, \{x,y\} \in E} w_2(\{x,y\}) \\times \exp(\\alpha * w(\{x,y\}))`.

    :See:

         Nishant Yadav, Ari Kobren, Nicholas Monath, Andrew Mccallum.
         `Supervised Hierarchical Clustering with Exponential Linkage <http://proceedings.mlr.press/v97/yadav19a.html>`_
         Proceedings of the 36th International Conference on Machine Learning, PMLR 97:6973-6983, 2019.

    :param graph: input graph
    :param edge_weights: edge weights of the input graph
    :param alpha: exponential parameter
    :param edge_weight_weights: weighting of edge weights of the input graph (default to an array of ones)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    alpha = float(alpha)

    if edge_weight_weights is None:
        edge_weight_weights = np.ones_like(edge_weights)
    else:
        edge_weights, edge_weight_weights = hg.cast_to_common_type(edge_weights, edge_weight_weights)

    # special cases: improve efficiency and avoid numerical issues
    if alpha == 0:
        tree, altitudes = hg.binary_partition_tree_average_linkage(graph, edge_weights, edge_weight_weights)
    elif alpha == float('-inf'):
        tree, altitudes = hg.binary_partition_tree_single_linkage(graph, edge_weights)
    elif alpha == float('inf'):
        tree, altitudes = hg.binary_partition_tree_complete_linkage(graph, edge_weights)
    else:
        res = hg.cpp._binary_partition_tree_exponential_linkage(graph, edge_weights, alpha, edge_weight_weights)
        tree = res.tree()
        altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def binary_partition_tree_single_linkage(graph, edge_weights):
    """
    Alias for :func:`~higra.bpt_canonical`.

    Given a graph :math:`G=(V, E)`, with initial edge weights :math:`w`,
    the distance :math:`d(X,Y)` between any two clusters :math:`X` and :math:`Y` is

    .. math::

        d(X,Y) = \min \{w(\{x,y\}) | x \in X, y \in Y, \{x,y\} \in E \}

    Regions are then iteratively merged following the above distance (closest first) until a single region remains.

    :param edge_weights: edge weights of the input graph (Concept :class:`~higra.CptEdgeWeightedGraph`)
    :param graph: input graph (deduced from :class:`~higra.CptEdgeWeightedGraph`)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes (Concept :class:`~higra.CptValuedHierarchy`)
    """

    return hg.bpt_canonical(graph, edge_weights)


def binary_partition_tree_ward_linkage(graph, vertex_centroids, vertex_sizes=None, altitude_correction="max"):
    """
    Binary partition tree with the Ward linkage rule.

    Given a graph :math:`G=(V, E)`, with initial edge weights :math:`w` with associated weights :math:`w'`,
    the distance :math:`d(X,Y)` between any two clusters :math:`X` and :math:`Y` is

    .. math::

        d(X,Y) = \\frac{| X |\\times| Y |}{| X |+| Y |} \| \\vec{X} - \\vec{Y} \|^2

    where :math:`\\vec{X}` and :math:`\\vec{Y}` are the centroids of  :math:`X` and  :math:`Y`.

    Regions are then iteratively merged following the above distance (closest first) until a single region remains

    Note that the Ward distance is not necessarily strictly increasing when processing a non complete graph.
    This can be corrected afterward with an altitude correction strategy. Valid values for ``altitude correction`` are:

        - ``"none"``: nothing is done and the altitude of a node is equal to the Ward distance between its 2 children;
          this may not be non-decreasing
        - ``"max"``: the altitude of a node :math:`n` is defined as the maximum of the the Ward distance associated
          to each node in the subtree rooted in :math:`n`.

    :param graph: input graph
    :param vertex_centroids: Centroids of the graph vertices (must be a 2d array)
    :param vertex_sizes: Size (number of elements) of the graph vertices (default to an array of ones)
    :param altitude_correction: can be ``"none"`` or ``"max"`` (default)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    if vertex_sizes is None:
        vertex_sizes = np.ones((graph.num_vertices(),), dtype=vertex_centroids.dtype)
    else:
        vertex_centroids, vertex_sizes = hg.cast_to_common_type(vertex_centroids, vertex_sizes)

    res = hg.cpp._binary_partition_tree_ward_linkage(graph, vertex_centroids, vertex_sizes, altitude_correction)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def binary_partition_tree(graph, weight_function, edge_weights):
    """
    Binary partition tree of the graph with a user provided cluster distance.

    At each step:

    1. the algorithm finds the edge of smallest weight;
    2. the two vertices linked by this edge are merged: the new vertex is the parent of the two merged vertices and its
       altitude is equal to the weight of the fusion edge;
    3. the weight of the edges linking the new vertex to the remaining vertices of the graph are updated according to
       the user provided function (:attr:`weight_function`);
    4. repeat until a single vertex remains

    The initial weight of the edges (:attr:`edge_weights`) and the callback (:attr:`weight_function`) determine the
    shape of the hierarchy. Note that the altitudes of the constructed hierarchy are not necessarily increasing;
    if needed this can be enforced as a post processing with the function :func:`~higra.tree_monotonic_regression`.

    Classical linkage functions are already implemented:

    - single/min linkage :func:`~higra.binary_partition_tree_single_linkage`
    - complete/max linkage :func:`~binary_partition_tree_complete_linkage`
    - average linkage :func:`~binary_partition_tree_average_linkage`
    - Ward linkage :func:`~binary_partition_tree_ward_linkage`.

    Those functions are implemented in c++ and should be faster than a user defined weighting function written in Python.

    :Weight function:

    The :attr:`weight_function` callback can be anything defining the operator ``()`` and should follow the
    following pattern:

    .. code-block:: python

        def weight_function(graph,              # the current state of the graph
                       fusion_edge_index,       # the edge between the two vertices being merged
                       new_region,              # the new vertex in the graph
                       merged_region1,          # the first vertex merged
                       merged_region2,          # the second vertex merged
                       new_neighbours):         # list of edges to be weighted (see below)
           ...
           for n in new_neighbours:
               ...
               n.set_new_edge_weight(new_edge_value) # define the weight of this edge

        }

    Each element in the parameter ``new_neighbours`` represent an edge between the new vertex and another vertex of
    the graph. For each element of the list, the following methods are available:

    - ``neighbour_vertex()``: the other vertex
    - ``num_edges()``: returns 2 if both the two merged vertices had an edge linking themselves with ``neighbour_vertex()`` and 1 otherwise
    - ``first_edge_index()``: the index of the edge linking one of the merged region to ``neighbour_vertex()``
    - ``second_edge_index()``: the index of the edge linking the other merged region to ``neighbour_vertex()`` (only if ``num_edges()==2``)
    - ``set_new_edge_weight(value)``: weight of the new edge. **This has to be defined in the weighting function**.
    - ``new_edge_index()``: the index of the new edge as the weighting function will probably have to track new weight values

    :Example:

    The following example shows how to define a weighting function for average linkage assuming that:

    - ``edge_weights`` is an array containing the weight of each edge, and
    - ``edge_counts`` is an array containing the number of edges present between two clusters: initially 1 for each edge but
      this will increase when clusters are merged.

    When a merge happens, there are two possibilities. Consider that we have three clusters :math:`A`, :math:`B`, and :math:`C` and
    we are merging :math:`A` and :math:`B` to obtain the new cluster :math:`D`.

    - If there is an edge of weight :math:`w_{AC}` and multiplicity :math:`c_{AC}` between :math:`A` and :math:`C` but
      there is no edge between :math:`B` and :math:`C`. Then, the new graph will contain an edge between :math:`D` and :math:`C`
      such that :math:`w_{DC}=w_{AC}` and :math:`c_{DC}=c_{AC}`. (The situation is similar if the edge were between
      :math:`B` and :math:`C`).
    - If there is an edge between :math:`A` and :math:`C` and between :math:`B` and :math:`C`. Then, the new graph will
      contain an edge between :math:`D` and :math:`C` such that :math:`w_{DC} = \\frac{w_{AC}*c_{AC} + w_{BC}*c_{BC}}{c_{AC} + c_{BC}}`
      and :math:`c_{DC}=c_{AC} + c_{BC}`.


    .. code-block:: python

        def weighting_function_average_linkage(graph, fusion_edge_index, new_region, merged_region1, merged_region2, new_neighbours):
            for n in new_neighbours:
                if n.num_edges() > 1:
                    new_count = edge_counts[n.first_edge_index()] + edge_counts[n.second_edge_index()]
                    new_weight = (edge_weights[n.first_edge_index()] * edge_counts[n.first_edge_index()] \\
                        + edge_weights[n.second_edge_index()] * edge_counts[n.second_edge_index()]) \\
                        / new_weight
                else:
                    new_count = edge_counts[n.first_edge_index()]
                    new_weight = edge_weights[n.first_edge_index()]

                n.set_new_edge_weight(new_weight)
                edge_weights[n.new_edge_index()] = new_weight
                edge_counts[n.new_edge_index()] = new_count

    :Complexity:

    The worst case time complexity is in :math:`\mathcal{O}(n^2\log(n))` with :math:`n` the number of vertices in the graph.
    The runtime complexity on a sparse graph with well structured data (far different from noise) can be much better in practice.

     .. warning::

        This function uses a Python callback (the :attr:`weight_function`) that is called frequently by the algorithm:
        performances will be far from optimal. Please consider a C++ implementation if it is too slow (see this
        `helper project <https://github.com/higra/Higra-cppextension-cookiecutter>`_ ).

    :param graph: input graph
    :param weight_function: see detailed description above
    :param edge_weights: edge weights of the input graph
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """
    res = hg.cpp._binary_partition_tree(graph, edge_weights, weight_function)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes
