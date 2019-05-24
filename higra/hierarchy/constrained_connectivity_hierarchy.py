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


def constrained_connectivity_hierarchy_alpha_omega(graph, vertex_weights):
    """
    Construct a alpha-omega constrained connectivity hierarchy based on the given vertex weighted graph.

    For  :math:`(i,j)` be an edge of the graph, we define :math:`w(i,j)=|w(i) - w(j)|`, the weight of this edge.
    Let :math:`X` be a set of vertices, the range of :math:`X` is the maximal absolute difference between the weights of any two vertices in :math:`X`:
    :math:`R(X) = \max\{|w(i) - w(j)|, (i,j)\in X^2\}`

    Let :math:`\\alpha` be a positive real number, a set of vertices :math:`X` is :math:`\\alpha`-connected, if for any two vertices
    :math:`i` and :math:`j` in :math:`X`, there exists a path from :math:`i` to :math:`j` in :math:`X` composed of edges of weights
    lower than or equal to :math:`\\alpha`.

    Let :math:`\\alpha`  and :math:`\omega` be a two positive real numbers, the :math:`\\alpha-\omega`-connected components of the graph are
    the maximal :math:`\\alpha'`-connected sets of vertices with a range lower than or equal to :math:`\omega`, with :math:`\\alpha'\leq\\alpha`.

    Finally, the alpha-omega constrained connectivity hierarchy is defined as the hierarchy composed of all the :math:`k-k`-connected components for all positive :math:`k`.

    The definition used follows the one given in:

        P. Soille,
        "Constrained connectivity for hierarchical image partitioning and simplification,"
        in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 30, no. 7, pp. 1132-1145, July 2008.
        doi: 10.1109/TPAMI.2007.70817

    The algorithm runs in time :math:`\mathcal{O}(n\log(n))` and proceeds by filtering a quasi-flat zone hierarchy (see :func:`~higra.quasi_flat_zones_hierarchy`)

    :param graph: input graph
    :param vertex_weights: edge_weights: edge weights of the input graph
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    vertex_weights = hg.linearize_vertex_weights(vertex_weights, graph)
    if vertex_weights.ndim != 1:
        raise ValueError("constrainted_connectivity_hierarchy_alpha_omega only works for scalar vertex weights.")

    # QFZ on the L1 distance weighted graph
    edge_weights = hg.weight_graph(graph, vertex_weights, hg.WeightFunction.L1)
    tree, altitudes = hg.quasi_flat_zones_hierarchy(graph, edge_weights)
    altitude_parents = altitudes[tree.parents()]

    # vertex value range inside each region
    min_value = hg.accumulate_sequential(tree, vertex_weights, hg.Accumulators.min)
    max_value = hg.accumulate_sequential(tree, vertex_weights, hg.Accumulators.max)
    value_range = max_value - min_value

    # parent node can't be deleted
    altitude_parents[tree.root()] = max(altitudes[tree.root()], value_range[tree.root()])

    # nodes whith a range greater than the altitudes of their parent have to be deleted
    violated_constraints = value_range >= altitude_parents

    # the altitude of nodes with a range greater than their altitude but lower than the one of their parent must be changed
    reparable_node_indices = np.nonzero(np.logical_and(value_range > altitudes, value_range < altitude_parents))
    altitudes[reparable_node_indices] = value_range[reparable_node_indices]

    # final  result construction
    tree, node_map = hg.simplify_tree(tree, violated_constraints)
    altitudes = altitudes[node_map]
    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def constrained_connectivity_hierarchy_strong_connection(graph, edge_weights):
    """
    Construct a strongly constrained connectivity hierarchy based on the given edge weighted graph.

    Let :math:`X` be a set of vertices, the range of :math:`X` is the maximal weight of the edges linking two vertices inside :math:`X`.

    Let :math:`\\alpha` be a positive real number, a set of vertices :math:`X` is :math:`\\alpha`-connected, if for any two vertices
    :math:`i` and :math:`j` in :math:`X`, there exists a path from :math:`i` to :math:`j` in :math:`X` composed of edges of weights
    lower than or equal to :math:`\\alpha`.

    Let :math:`\\alpha`  be a positive real numbers, the :math:`\\alpha`-strongly connected components of the graph are
    the maximal :math:`\\alpha'`-connected sets of vertices with a range lower than or equal to :math:`\\alpha` with :math:`\\alpha'\leq\\alpha`.

    Finally, the strongly constrained connectivity hierarchy is defined as the hierarchy composed of all the
    :math:`\\alpha`- strongly connected components for all positive :math:`\\alpha`.

    The definition used follows the one given in:

        P. Soille,
        "Constrained connectivity for hierarchical image partitioning and simplification,"
        in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 30, no. 7, pp. 1132-1145, July 2008.
        doi: 10.1109/TPAMI.2007.70817

    The algorithm runs in time :math:`\mathcal{O}(n\log(n))` and proceeds by filtering a quasi-flat zone hierarchy (see :func:`~higra.quasi_flat_zones_hierarchy`)

    :param graph: input graph
    :param vertex_weights: edge_weights: edge weights of the input graph
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    tree, altitudes = hg.quasi_flat_zones_hierarchy(graph, edge_weights)
    altitude_parents = altitudes[tree.parents()]

    # max edge weights inside each region
    lca_map = hg.attribute_lca_map(tree)
    max_edge_weights = np.zeros((tree.num_vertices(),), dtype=edge_weights.dtype)
    np.maximum.at(max_edge_weights, lca_map, edge_weights)
    max_edge_weights = hg.accumulate_and_max_sequential(tree,
                                                        max_edge_weights,
                                                        max_edge_weights[:tree.num_leaves()],
                                                        hg.Accumulators.max)

    # parent node can't be deleted
    altitude_parents[tree.root()] = max(altitudes[tree.root()], max_edge_weights[tree.root()])

    # nodes whith a range greater than the altitudes of their parent have to be deleted
    violated_constraints = max_edge_weights >= altitude_parents

    # the altitude of nodes with a range greater than their altitude but lower than the one of their parent must be changed
    reparable_node_indices = np.nonzero(
        np.logical_and(max_edge_weights > altitudes, max_edge_weights < altitude_parents))
    altitudes[reparable_node_indices] = max_edge_weights[reparable_node_indices]

    # final  result construction
    tree, node_map = hg.simplify_tree(tree, violated_constraints)
    altitudes = altitudes[node_map]
    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes
