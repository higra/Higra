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


@hg.argument_helper(hg.CptVertexLabeledGraph)
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

    :param vertex_labels: labeling of the graph vertices into super-vertices (Concept :class:`~higra.CptVertexLabeledGraph`)
    :param other_hierarchies: a hierarchy or a list of hierarchies: hierarchies can be given either as trees (Concept :class:`~higra.CptValuedHierarchy`) or as saliency maps (Concept :class:`~higra.CptSaliencyMap`), defined on the pixel graph or on a region adjacency graph (Concept :class:`~higra.CptRegionAdjacencyGraph`).
    :param graph: the domain graph (deduced from :class:`~higra.CptVertexLabeledGraph`)
    :return: a hierarchy or a list of hierarchies as saliency maps (Concept :class:`~higra.CptSaliencyMap`).
    """
    result = []
    list_input = True
    if type(other_hierarchies) is np.ndarray:
        list_input = False
        other_hierarchies = (other_hierarchies,)
    else:
        try:
            _ = iter(other_hierarchies)
        except TypeError:  # other_hierachies is not iterable
            list_input = False
            other_hierarchies = (other_hierarchies,)

    aligner = hg.HierarchyAligner.from_labelisation(graph, vertex_labels)

    for hierarchy in other_hierarchies:
        r = None

        if hg.CptValuedHierarchy.validate(hierarchy):
            h = hg.CptValuedHierarchy.construct(hierarchy)

            if hg.CptRegionAdjacencyGraph.validate(h["leaf_graph"]):
                rag = hg.CptRegionAdjacencyGraph.construct(h["leaf_graph"])
                r = aligner.align_hierarchy(rag["vertex_map"], h["tree"], h["altitudes"])
            else:
                r = aligner.align_hierarchy(h["tree"], h["altitudes"])

        elif hg.CptSaliencyMap.validate(hierarchy):
            h = hg.CptSaliencyMap.construct(hierarchy)
            if hg.CptRegionAdjacencyGraph.validate(h["graph"]):
                rag = hg.CptRegionAdjacencyGraph.construct(h["graph"])
                bpt, altitudes = hg.bpt_canonical(h["edge_weights"])
                r = aligner.align_hierarchy(rag["vertex_map"], bpt, altitudes)
            else:
                r = aligner.align_hierarchy(h["graph"], h["edge_weights"])

        else:
            raise Exception("Hierarchy format not recognized: " + str(hierarchy))

        hg.CptSaliencyMap.link(r, graph)
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
