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
    Seeds are defined as vertex weights: any flat zone of value strictly greater than 0 is considered as a seed.

    Note that if two different seeds are places in a minima of the edge weighted graph, and if the altitude of this minima
    is equal to the smallest representable value for the given `dtype` of the edge weights, then the algorithm won't be able
    to produce two different regions for these two seeds.

    :param graph: input graph
    :param edge_weights: Weights on the edges of the graph
    :param vertex_seeds: Seeds on the vertices of the graph
    :return: A labelisation of the graph vertices
    """
    # edges inside a seed take the value of the seed and 0 otherwise
    edges_in_or_between_seeds = hg.weight_graph(graph, vertex_seeds, hg.WeightFunction.L0)
    edges_outside_seeds = hg.weight_graph(graph, vertex_seeds, hg.WeightFunction.min)
    edges_in_seed = np.logical_and(edges_outside_seeds > 0, 1 - edges_in_or_between_seeds)

    # set edges inside seeds at minimum level
    edge_weights = edge_weights.copy()
    edge_weights[edges_in_seed > 0] = hg.dtype_info(edge_weights.dtype).min

    tree, altitudes = hg.watershed_hierarchy_by_attribute(
        graph,
        edge_weights,
        lambda tree, _: hg.accumulate_sequential(tree, vertex_seeds, hg.Accumulators.max))

    return hg.labelisation_hierarchy_supervertices(tree, altitudes)
