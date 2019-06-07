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


def align_hierarchies(graph, vertex_labels, other_hierarchies):
    """
    Align hierarchies boundaries on the boundaries of the provided super-vertex decomposition of a graph

    Given:

        - a graph :math:`G=(V,E)`
        - a fine labelisation :math:`l_1` of the vertices of :math:`G`;
        - a tree :math:`T` on :math:`G` whose supervertices corresponds to the coarse labelisation :math:`l_2`
          of the vertices of :math:`G`; and
        - the altitudes :math:`a` of the nodes of :math:`T`.

    Let us denote:

        - given a vertex :math:`x` of :math:`G` and a labelisation :math:`l`, :math:`l(x)` is the region of :math:`l`
          that contains :math:`x`
        - given a region :math:`r` of :math:`l_1`, :math:`s(r, l_2)` is the region :math:`R` of :math:`l_2` that has
          the largest intersection with :math:`r`, ie, :math:`s(r, l_2) = \\arg \max_{R \in l_2} | R \cap r |`

    The projection of :math:`T` onto :math:`l_1` is a hierarchy given by the saliency map :math:`sm` on :math:`G`
    defined by:

    .. math::

        \\forall e_{xy} \in E, sm(e_{xy}) = a(lca_T(s(l_1(x), l_2), s(l_1(y), l_2)))

    where :math:`lca_T(x, y)` is the lowest common ancestor of nodes :math:`x` and :math:`y` in :math:`T`.

    :param graph: the domain graph
    :param vertex_labels: 1d array of positive integers, labeling of the graph vertices into super-vertices
    :param other_hierarchies: a hierarchy or a list of hierarchies: hierarchies can be given either as valued trees
        (pairs (tree, altitudes) ) or as saliency maps (pairs (graph, edge_weights)), defined on the pixel graph or
        on a region adjacency graph (Concept :class:`~higra.CptRegionAdjacencyGraph`).
    :return: a hierarchy or a list of hierarchies as saliency maps
    """
    result = []
    list_input = True
    if not hg.is_iterable(other_hierarchies):
        raise TypeError("bas format for other hierarchies.")

    first_element = other_hierarchies[0]
    if not hg.is_iterable(first_element):
        list_input = False
        other_hierarchies = (other_hierarchies,)

    aligner = hg.HierarchyAligner.from_labelisation(graph, vertex_labels)

    for hierarchy in other_hierarchies:
        obj, values = hierarchy
        if type(obj) is hg.Tree:
            leaf_graph = hg.CptHierarchy.get_leaf_graph(obj)
            if leaf_graph is not None and hg.CptRegionAdjacencyGraph.validate(leaf_graph):
                vertex_map = hg.CptRegionAdjacencyGraph.get_vertex_map(leaf_graph)
                r = aligner.align_hierarchy(vertex_map, obj, values)
            else:
                r = aligner.align_hierarchy(obj, values)

        elif type(obj) is hg.UndirectedGraph:
            if hg.CptRegionAdjacencyGraph.validate(obj):
                vertex_map = hg.CptRegionAdjacencyGraph.get_vertex_map(obj)
                bpt, altitudes = hg.bpt_canonical(obj, values)
                r = aligner.align_hierarchy(vertex_map, bpt, altitudes)
            else:
                r = aligner.align_hierarchy(obj, values)

        else:
            raise Exception("Hierarchy format not recognized: " + str(hierarchy))
        result.append(r)
    if not list_input:
        return result[0]
    return result


def project_fine_to_coarse_rag(fine_rag, coarse_rag):
    """
    Find for each region of the fine rag, the region of the
    coarse rag that maximises the intersection with the "fine" region.

    See: :func:`~higra.project_fine_to_coarse_labelisation`

    :param fine_rag: reference region adjacency graph (Concept :class:`~higra.CptRegionAdjacencyGraph`)
    :param coarse_rag: region adjacency graph to align (Concept :class:`~higra.CptRegionAdjacencyGraph`)
    :return: a 1d array of size ``fine_rag.num_vertices()``
    """
    return hg.project_fine_to_coarse_labelisation(
        hg.get_attribute(fine_rag, "vertex_map"),
        hg.get_attribute(coarse_rag, "vertex_map"),
        fine_rag.num_vertices(),
        coarse_rag.num_vertices())


def project_fine_to_coarse_labelisation(labelisation_fine, labelisation_coarse, num_regions_fine=0, num_regions_coarse=0):
    """
    Find for each label (ie region) of the fine labelisation, the label of the region in the
    coarse labelisation that maximises the intersection with the fine region.

    Pre-condition:

        - :math:`range(labelisation\_fine) = [0, \ldots, num\_regions\_fine[`
        - :math:`range(labelisation\_coarse) = [0, \ldots, num\_regions\_coarse[`

    Then, for all label :math:`i \in [0, \ldots, num\_regions\_fine[`,

    .. math::

        result(i) = \\arg \max_{j \in [0, \ldots, num\_regions\_coarse[} | (fine\_labelisation == i) \cap (coarse\_labelisation == j) |


    If :attr:`num_regions_fine` or :attr:`num_regions_coarse` are not provided, they will be determined as
    :math:`max(labelisation\_fine) + 1` and  :math:`max(labelisation\_coarse) + 1`

    :param labelisation_fine: 1d array of positive integers
    :param labelisation_coarse: 1d array of positive integers (same length as :attr:`labelisation_fine`)
    :param num_regions_fine: optional, number of different labels in :attr:`labelisation_fine`
    :param num_regions_coarse: optional, number of different labels in :attr:`labelisation_coarse`
    :return: a 1d array mapping fine labels to coarse labels
    """
    labelisation_fine, labelisation_coarse = hg.cast_to_common_type(labelisation_fine, labelisation_coarse)

    return hg.cpp._project_fine_to_coarse_labelisation(labelisation_fine, labelisation_coarse, num_regions_fine, num_regions_coarse)
