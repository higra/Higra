############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Wonder Alexandre Luz Alves                              #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import higra as hg


def _get_attribute_map():
    return {
        "area": hg.CasfAttribute.area,
        "bounding_box_width": hg.CasfAttribute.bounding_box_width,
        "bounding_box_height": hg.CasfAttribute.bounding_box_height,
        "bounding_box_diagonal": hg.CasfAttribute.bounding_box_diagonal,
    }


def connected_alternating_sequential_filter(graph, vertex_weights, attribute, thresholds):
    r"""
    Connected alternating sequential filter (CASF) on a vertex weighted graph.

    Let :math:`G=(V,E)` be the input graph, :math:`f_0:V\to K` the input
    vertex weights, and :math:`A` an increasing attribute on the component
    trees. For a threshold :math:`t`, denote by :math:`\gamma^A_t` the
    connected attribute opening obtained by pruning max-tree nodes whose
    attribute value is less than or equal to :math:`t`, and by
    :math:`\phi^A_t` the corresponding connected attribute closing obtained
    by pruning min-tree nodes. Given the threshold sequence
    :math:`T=(t_1,\ldots,t_k)`, this implementation computes

    .. math::

        f_i = \phi^A_{t_i}\!\left(\gamma^A_{t_i}(f_{i-1})\right),
        \quad i=1,\ldots,k,

    and returns :math:`f_k`. Thus, each threshold applies the anti-extensive
    attribute opening on the max-tree first, followed by the extensive
    attribute closing on the min-tree. Thresholds are applied in the order
    given; a standard alternating sequential filter uses a non-decreasing
    sequence.

    The max-tree and min-tree are constructed only once. When one tree is
    pruned, its dual tree and both attributes are updated incrementally, and
    the filtered vertex weights are reconstructed only after the complete
    sequence. The update strategy and its complexity analysis are described
    in [AlvesEtAl2025]_, [AlvesEtAl2026]_.

    :Complexity:

    For bounded-degree image grids and the comparison-based component tree
    construction analyzed in [AlvesEtAl2026]_, rebuilding a tree at every opening and
    closing step has time complexity
    :math:`\mathcal{O}(k |V| \log |V|)`. Let :math:`S_i` denote the roots of
    the subtrees pruned at iteration :math:`i`, :math:`|C_a|` the size of an
    affected component, and :math:`|b-a|` the altitude interval involved in
    the update of its dual tree. Under the same assumptions, the incremental
    algorithm has time complexity

    .. math::

        \mathcal{O}\!\left(
            |V|\log |V| +
            \sum_{i=1}^{k}\sum_{C\in S_i} (|C_a| + |b-a|)
        \right).

    It is therefore expected to outperform the naive approach when the local
    update work is smaller than rebuilding the full trees. The gain is usually
    more pronounced for longer threshold sequences, after the one-time tree
    construction cost has been amortized. Early iterations that remove large
    components can be less favorable. For a full CASF run with :math:`k=50`
    thresholds, the experiments reported in [AlvesEtAl2026]_ show speed-ups of
    :math:`20.452\times`, :math:`20.643\times`, and :math:`20.572\times` for
    the update-based implementation over the naive baseline on the 480p,
    720p, and 1080p versions of the Occluded RoadText Challenge dataset,
    respectively. These measured speed-ups are indicative rather than
    guaranteed: actual gains also depend on the input vertex weights,
    attribute, graph degree, altitude data type, and threshold sequence.

    :Example:

    .. code-block:: python

        import higra as hg
        import numpy as np

        image = np.asarray(((0, 0, 0, 0, 0),
                            (0, 7, 7, 0, 0),
                            (0, 7, 7, 0, 5),
                            (0, 0, 0, 0, 5),
                            (0, 0, 0, 0, 0)), dtype=np.uint8)
        graph = hg.get_4_adjacency_implicit_graph(image.shape)
        filtered = hg.connected_alternating_sequential_filter(
            graph, image, "area", thresholds=[1, 2]
        )

    .. [AlvesEtAl2025] W. A. L. Alves, N. Passat, D. J. Silva, A. Morimitsu, and R. F. Hashimoto,
       `Efficient Connected Alternating Sequential Filters Based on Component Trees
       <https://doi.org/10.1007/978-3-032-09544-2_15>`_, DGMM 2025, Lecture Notes in
       Computer Science, pp. 210-223.
    .. [AlvesEtAl2026] W. A. L. Alves, N. Passat, D. J. Silva, A. Morimitsu, and R. F. Hashimoto,
       `Component Tree: Update Rather than Rebuild
       <https://doi.org/10.1007/s10851-026-01321-w>`_, Journal of Mathematical Imaging
       and Vision, vol. 68, article 61, 2026.

    :param graph: input graph
    :param vertex_weights: vertex weights of the input graph
    :param attribute: one of ``"area"``, ``"bounding_box_width"``,
        ``"bounding_box_height"``, ``"bounding_box_diagonal"``, or the
        corresponding :class:`~higra.CasfAttribute` value. Bounding-box
        attributes require a graph with a two-dimensional embedding.
    :param thresholds: sequence of thresholds applied successively. Use a
        non-decreasing sequence to obtain a standard CASF.
    :return: filtered vertex weights with the same shape as ``vertex_weights``
    """
    attribute_map = _get_attribute_map()

    if isinstance(attribute, str):
        try:
            attribute = attribute_map[attribute]
        except KeyError:
            raise ValueError("Unknown CASF attribute '" + attribute + "'. Expected one of " + str(tuple(attribute_map.keys())) + ".")
    elif attribute not in attribute_map.values():
        raise ValueError("attribute must be a string or a higra.CasfAttribute value.")

    vertex_weights = hg.linearize_vertex_weights(vertex_weights, graph)

    filtered_weights = hg.cpp._connected_alternating_sequential_filter(graph, vertex_weights, attribute, thresholds)

    return hg.delinearize_vertex_weights(filtered_weights, graph)
