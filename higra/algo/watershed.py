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


def labelisation_seeded_watershed(graph, edge_weights, vertex_seeds, background_label=0):
    """
    Seeded watershed cut on an edge weighted graph.
    Seeds and associated labels are given in :attr:`vertex_seeds`.
    A vertex :math:`v`, such that :math:`vertex\_seeds(v)\\neq background\_label` is a seed with associated label :math:`vertex\_seeds(v)`.

    The label of a vertex of the graph is then defined equal to the label of the closest seed in the edge weighted graph for the min-max distance.
    If several such seeds exist (eg. on a plateus between two seeds), an arbitrary and consistent choice is made ensuring that:

    - each flat zone of level :math:`k` of the final labelling contains at least one seed with the label :math:`k`; and
    - each seed is contained in a flat zone whose level is equal to the seed label.

    :Complexity:

    This algorithm has a runtime complexity in :math:`\mathcal{O}(n \log n)` with :math:`n` the number of edges in the graph.

    :param graph: Input graph
    :param edge_weights: Weights on the edges of the graph
    :param vertex_seeds: Seeds with integer label values on the vertices of the graph
    :param background_label: Vertices whose values are equal to :attr:`background_label` (default 0) in :attr:`vertex_seeds` are not considered as seeds
    :return: A labelisation of the graph vertices
    """
    if not issubclass(vertex_seeds.dtype.type, np.integer):
        raise ValueError("vertex_seeds must be an array of integers")

    vertex_seeds = hg.linearize_vertex_weights(vertex_seeds, graph)

    vertex_seeds = hg.cast_to_dtype(vertex_seeds, np.int64)

    labels = hg.cpp._labelisation_seeded_watershed(graph, edge_weights, vertex_seeds, background_label)

    labels = hg.delinearize_vertex_weights(labels, graph)
    return labels
