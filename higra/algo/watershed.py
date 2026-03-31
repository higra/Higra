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


class IncrementalWatershedCut:
    """
    Incremental seeded watershed cut based on the binary partition tree.

    This class provides an efficient way to compute seeded watershed cuts
    in an interactive segmentation setting, where seeds are added and removed
    incrementally. Instead of recomputing the full watershed from scratch at
    each interaction, only the affected regions are updated.

    The algorithm maintains a canonical BPT and a visitCount array to identify
    watershed edges. The labeling is obtained by BFS on the MST forest.

    Reference:

        Q. Lebon, J. Lefevre, J. Cousty, B. Perret.
        `Interactive Segmentation With Incremental Watershed Cuts <https://hal.science/hal-04069187v1>`_.
        CIARP 2023.

    :param graph: input graph (must be connected)
    :param edge_weights: Weights on the edges of the graph
    """

    def __init__(self, graph, edge_weights):
        tree, _ = hg.bpt_canonical(graph, edge_weights, compute_mst=True)
        mst = hg.CptBinaryHierarchy.get_mst(tree)
        self._impl = hg.cpp.IncrementalWatershedCut(tree, mst)
        self._graph = graph

    def add_seeds(self, seed_vertices, seed_labels):
        """
        Add seeds to the current watershed cut.

        Each seed is defined by a vertex index and a label. Two seeds cannot share
        the same vertex but can share the same label (resulting in merged regions
        in the output labeling). Labels must be non-zero (0 is reserved for
        unlabeled/background vertices).

        :param seed_vertices: 1d array of seed vertex indices
        :param seed_labels: 1d array of seed labels (same size as seed_vertices)
        """
        seed_vertices = np.asarray(seed_vertices, dtype=np.int64).ravel()
        seed_labels = np.asarray(seed_labels, dtype=np.int64).ravel()
        self._impl._add_seeds(seed_vertices, seed_labels)

    def remove_seeds(self, seed_vertices):
        """
        Remove seeds from the current watershed cut.

        :param seed_vertices: 1d array of seed vertex indices to remove
        """
        seed_vertices = np.asarray(seed_vertices, dtype=np.int64).ravel()
        self._impl._remove_seeds(seed_vertices)

    def get_labeling(self):
        """
        Compute and return the current vertex labeling.

        Vertices with no seed in their component are labeled 0 (background).

        :return: A labeling of the graph vertices
        """
        labels = self._impl._get_labeling()
        labels = hg.delinearize_vertex_weights(labels, self._graph)
        return labels
