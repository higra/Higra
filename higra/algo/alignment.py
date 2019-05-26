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
        - a graph g
        - a fine labelisation l1 of the vertices of g;
        - a tree t on g whose supervertices corresponds to the coarse labelisation l2 of the vertices of g; and
        - the altitudes a of the nodes of t.
    Let us denote:
        - given a vertex x of g and a labelisation l, l(x) is the region of l that contains x
        - given a region r of l1, s(r, l2) is the region R of l2 that has the largest intersection with r, ie, s(r, l2) = arg_max(R in l2) #(R \cap r)
    The projection of t onto l1 is a hierarchy given by the saliency map sm on g defined by:
           for all {x,y} in edges(g), sm({x,y}) = a(lca_t(s(l1(x), l2), s(l1(y), l2)))

    :param graph: the domain graph
    :param vertex_labels: labeling of the graph vertices into super-vertices
    :param other_hierarchies: a hierarchy or a list of hierarchies: hierarchies can be given either as valued trees (pairs (tree, altitudes) ) or as saliency maps (pairs (graph, edge_weights)), defined on the pixel graph or on a region adjacency graph (Concept :class:`~higra.CptRegionAdjacencyGraph`).
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
    Given two region adjacency graphs, a fine and a coarse one, of a same set of elements.
    Find for each region of the fine rag, the region of the
    coarse rag that maximises the intersection with the "fine" region.

    :param fine_rag: reference region adjacency graph (Concept :class:`~higra.CptRegionAdjacencyGraph`)
    :param coarse_rag: region adjacency graph to align (Concept :class:`~higra.CptRegionAdjacencyGraph`)
    :return: a 1d array of size num_vertices fine_rag.num_vertices()
    """
    return hg.project_fine_to_coarse_labelisation(
        hg.get_attribute(fine_rag, "vertex_map"),
        hg.get_attribute(coarse_rag, "vertex_map"),
        fine_rag.num_vertices(),
        coarse_rag.num_vertices())
