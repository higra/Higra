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


def labelisation_watershed(graph, edge_weights):
    """
    Watershed cut of the given edge weighted graph.

    The definition and algorithm used are described in:

        J. Cousty, G. Bertrand, L. Najman and M. Couprie.
        `Watershed cuts: minimum spanning forests, and the drop of water principle <https://hal-upec-upem.archives-ouvertes.fr/hal-00622410/document>`_.
        IEEE Trans. on Pattern Analysis and Machine Intelligence, 31(8): 1362-1374, 2009.

    The watershed cut is represented by a labelisation of the graph vertices.

    :Complexity:

    This algorithm has a linear runtime complexity :math:`\mathcal{O}(n)` with :math:`n` the number of edges in the graph.


    :param graph: input graph
    :param edge_weights: Weights on the edges of the graph
    :return: A labelisation of the graph vertices
   """
    vertex_labels = hg.cpp._labelisation_watershed(graph, edge_weights)

    vertex_labels = hg.delinearize_vertex_weights(vertex_labels, graph)

    return vertex_labels


def labelisation_seeded_watershed(graph, edge_weights, vertex_seeds):
    """
    Seeded watershed cut on an edge weighted graph.
    Seeds are defined as integral vertex weights: vertices with non zero values are considered as seeds.
    The label of a vertex of the graph is then defined equal to the label of the closest seed in the edge weighted graph for the min-max distance.
    If several such seeds exist (eg. on a plateus between two seeds), an arbitrary and consistent choice is made.

    The seeds must be connected: for any seed value :math:`v`, the set of vertices of the input graph having the seed value
    :math:`v` must be connected in the input graph.

    :Complexity:

    This algorithm has a runtime complexity in :math:`\mathcal{O}(n \log n)` with :math:`n` the number of edges in the graph.

    :param graph: input graph
    :param edge_weights: Weights on the edges of the graph
    :param vertex_seeds: Seeds on the vertices of the graph, seeds must be connected
    :return: A labelisation of the graph vertices
    """
    if not issubclass(vertex_seeds.dtype.type, np.integer):
        raise ValueError("vertex_seeds must be an array of integers")

    # identify edges inside seeds to enforce existence of minima
    edges_border_seeds = hg.weight_graph(graph, vertex_seeds, hg.WeightFunction.L0)
    edges_not_border_seeds = np.logical_not(edges_border_seeds)
    edges_inside_or_between_seeds = hg.weight_graph(graph, vertex_seeds, hg.WeightFunction.min) > 0
    edges_inside_seeds = np.logical_and(edges_inside_or_between_seeds, edges_not_border_seeds)

    # set edges inside seeds at minimum level
    edge_weights = edge_weights.copy()
    edge_weights[edges_inside_seeds > 0] = hg.dtype_info(edge_weights.dtype).min

    # compute mst and bpt
    tree, altitudes = hg.bpt_canonical(graph, edge_weights)

    # identify watershed edges as nodes where different seed labels merge
    label_tree = hg.accumulate_sequential(tree, vertex_seeds, hg.Accumulators.max)
    label_left_children = label_tree[tree.child(0)]
    label_right_children = label_tree[tree.child(1)]
    watershed_edges = np.logical_and(label_left_children != 0,
                                     np.logical_and(label_right_children != 0,
                                                    label_left_children != label_right_children))

    # the watershed edges form a cut in the mst, we take the corresponding labeliasation
    mst = hg.CptBinaryHierarchy.get_mst(tree)
    labels = hg.graph_cut_2_labelisation(mst, watershed_edges)
    labels = hg.delinearize_vertex_weights(labels, graph)

    # we ensure that final labels correspond to initial seed labels:
    # each region takes the label of the seed it contains.
    flat_labels = labels.flatten()
    flat_seeds = vertex_seeds.flatten()
    seed_positions, = np.nonzero(flat_seeds)
    seed_values = flat_seeds[seed_positions]
    label_values = flat_labels[seed_positions]
    representative_label_values, representative_label_positions = np.unique(label_values, return_index=True)
    representative_seed_values = seed_values[representative_label_positions]
    labels = representative_seed_values[labels - 1]

    return labels
