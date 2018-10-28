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


@hg.data_consumer(graph="domain")
def align_hierarchies(vertex_labels, other_hierarchies, graph):
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

    :param vertex_labels: labeling of the graph vertices into super-vertices
    :param other_hierarchies: a hierarchy or a list of hierarchies: hierarchies can be given either as trees or as saliency maps, defined on the pixel graph or on a region adjacency graph.
    :param graph: the domain graph
    :return: a hierarchy or a list of hierarchies as saliency maps.
    """
    result = []
    try:
        _ = iter(other_hierarchies)
    except TypeError: # other_hierachies is not iterable
        other_hierarchies = (other_hierarchies, )

    aligner = hg.HierarchyAligner.from_labelisation(graph, vertex_labels)

    for hierarchy in other_hierarchies:
        r = None
        if type(hierarchy) is hg.Tree:
            super_vertices=None
            leaf_graph = hg.get_attribute(hierarchy, "leaf_graph")
            if leaf_graph:
                super_vertices = hg.get_attribute(leaf_graph, "vertex_map")
            if super_vertices: #tree on rag
                r = aligner.align_hierarchy(super_vertices, hierarchy, hg.get_attribute(hierarchy, "altitudes"))
            else:
                r = aligner.align_hierarchy(hierarchy, hg.get_attribute(hierarchy, "altitudes"))
        else: # saliency map
            super_vertices = hg.get_attribute(hierarchy, "vertex_map")
            if super_vertices: # defined on rag
                bpt = hg.bpt_canonical(hierarchy)
                r = aligner.align_hierarchy(super_vertices, bpt, hg.get_attribute(bpt, "altitudes"))
            else:
                r = aligner.align_hierarchy(hierarchy, hg.get_attribute(hierarchy, "edge_weights"))
        hg.set_attribute(r, "domain", graph)
        result.append(r)
    if len(result) == 1:
        return result[0]
    return result


def project_fine_to_coarse_rag(fine_rag, coarse_rag):
    """
    Given two region adjacency graphs, a fine and a coarse one, of a same set of elements.
    Find for each region of the fine rag, the region of the
    coarse rag that maximises the intersection with the "fine" region.

    :param fine_rag:
    :param coarse_rag:
    :return: a 1d array of size num_vertices fine_rag.num_vertices()
    """
    return hg.project_fine_to_coarse_labelisation(
        hg.get_attribute(fine_rag, "vertex_map"),
        hg.get_attribute(coarse_rag, "vertex_map"),
        fine_rag.num_vertices(),
        coarse_rag.num_vertices())
