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

    :Complexity:

    Construction is dominated by the canonical BPT and MST construction in
    :math:`\\mathcal{O}(n \\log n)` with :math:`n` the number of edges in the
    input graph (dominated by the edge sort).

    :func:`add_seeds` on a batch of :math:`K` seeds runs in
    :math:`\\mathcal{O}(K \\cdot d + S)` where :math:`d` is the height of the
    BPT (:math:`\\mathcal{O}(\\log N)` for balanced trees, :math:`\\mathcal{O}(N)`
    worst case, with :math:`N` the number of vertices) and :math:`S` is the
    total size of the relabeled components.

    :func:`remove_seeds` on a batch of :math:`K` seeds runs in
    :math:`\\mathcal{O}(K \\cdot d + S)` same as :func:`add_seeds`.

    :func:`get_labeling` is :math:`\\mathcal{O}(1)` (the labeling is
    maintained incrementally).

    If Higra is built with parallel support (TBB), the relabeling performed by
    :func:`add_seeds` and :func:`remove_seeds` can use several threads. The result
    does not depend on the number of threads. See :func:`set_max_concurrency`,
    :func:`set_minimum_parallel_front_size` and :func:`set_num_parallel_hops`.

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

    def set_minimum_parallel_front_size(self, minimum_parallel_front_size):
        """
        Set the minimum size of the propagation front for parallel relabeling.

        When seeds are added or removed, labels are propagated from a front of vertices.
        The propagation is distributed between threads only while the front contains more
        than :attr:`minimum_parallel_front_size` vertices; smaller fronts are processed
        sequentially. The default value is 200.

        If Higra was built without parallel support (TBB), this function has no effect
        and logs a warning.

        :param minimum_parallel_front_size: minimum number of vertices in the front (positive integer)
        :raises ValueError: if :attr:`minimum_parallel_front_size` is not positive
        """
        if minimum_parallel_front_size <= 0:
            raise ValueError("minimum_parallel_front_size must be positive.")

        self._impl._set_minimum_parallel_front_size(minimum_parallel_front_size)

    def get_minimum_parallel_front_size(self):
        """
        Minimum size of the propagation front for parallel relabeling.

        See :func:`set_minimum_parallel_front_size`.

        :return: a positive integer
        """
        return self._impl._get_minimum_parallel_front_size()

    def set_num_parallel_hops(self, num_parallel_hops):
        """
        Set the number of propagation steps that each thread performs on its part of the
        front before the fronts of all threads are merged and divided again.

        Larger values reduce synchronization between threads but may balance the work less
        evenly. The default value is 10.

        If Higra was built without parallel support (TBB), this function has no effect
        and logs a warning.

        :param num_parallel_hops: number of independent propagation steps (positive integer)
        :raises ValueError: if :attr:`num_parallel_hops` is not positive
        """
        if num_parallel_hops <= 0:
            raise ValueError("num_parallel_hops must be positive.")

        self._impl._set_num_parallel_hops(num_parallel_hops)

    def get_num_parallel_hops(self):
        """
        Number of propagation steps that each thread performs independently.

        See :func:`set_num_parallel_hops`.

        :return: a positive integer
        """
        return self._impl._get_num_parallel_hops()

    def set_max_concurrency(self, maximum_concurrency):
        """
        Set the maximum number of threads used to relabel the vertices.

        A value of 1 disables parallel relabeling. Values larger than the number of
        threads available to Higra are reduced to that number, which is also the
        default value.

        If Higra was built without parallel support (TBB), this function has no effect
        and logs a warning: relabeling is always sequential.

        :param maximum_concurrency: maximum number of threads (positive integer)
        :raises ValueError: if :attr:`maximum_concurrency` is not positive
        """
        if maximum_concurrency <= 0:
            raise ValueError("maximum_concurrency must be positive.")

        self._impl._set_max_concurrency(maximum_concurrency)

    def get_max_concurrency(self):
        """
        Maximum number of threads used to relabel the vertices.

        See :func:`set_max_concurrency`. Always 1 if Higra was built without parallel
        support (TBB).

        :return: a positive integer
        """
        return self._impl._get_max_concurrency()

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
        if seed_vertices.size != seed_labels.size:
            raise ValueError("seed_vertices and seed_labels must have the same size.")

        # assert that seed labels are non-zero
        if np.any(seed_labels == 0):
            raise ValueError("seed_labels must be non-zero (0 is reserved for background).")

        # assert that seed vertices are valid indices
        if np.any(seed_vertices < 0) or np.any(seed_vertices >= self._graph.num_vertices()):
            raise ValueError("seed_vertices must be valid vertex indices in the graph.")

        self._impl._add_seeds(seed_vertices, seed_labels)

    def remove_seeds(self, seed_vertices):
        """
        Remove seeds from the current watershed cut.

        :param seed_vertices: 1d array of seed vertex indices to remove
        """
        seed_vertices = np.asarray(seed_vertices, dtype=np.int64).ravel()
        if seed_vertices.size == 0:
            return

        # assert that seed vertices are valid indices
        if np.any(seed_vertices < 0) or np.any(seed_vertices >= self._graph.num_vertices()):
            raise ValueError("seed_vertices must be valid vertex indices in the graph.")

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
