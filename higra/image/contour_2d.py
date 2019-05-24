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


@hg.argument_helper(hg.CptRegionAdjacencyGraph, ("pre_graph", hg.CptGridGraph))
def rag_2d_vertex_perimeter_and_edge_length(rag,
                                            pre_graph,
                                            shape,
                                            epsilon=0.1,
                                            relative_epsilon=True,
                                            min_size=2):
    """
    Estimate the vertex perimeter and the length of the frontier associated to the edges of a
    region adjacency graph constructed on a 2d 4 adjacency graph.

    The region boundaries are simplified with Ramer–Douglas–Peucker algorithm and is controlled
    by the parameters epsilon, relative_epsilon, min_size.
    See function :func:`~higra.Contour2d.subdivide` for more information.

    :param rag: input region adjacency graph (Concept :class:`~higra.RegionAdjacencyGraph`)
    :param pre_graph: graph on which the rag is build (Deduced from :class:`~higra.RegionAdjacencyGraph`, Concept :class:`~higra.GridGraph`)
    :param shape: shape of the pre_graph (Deduced from :class:`~higra.GridGraph`)
    :param epsilon: larger epsilon values will provide stronger contour shapes simplification
    :param relative_epsilon: Is epsilon given in relative or absolute units
    :param min_size: Boundaries elements smaller than min_size will be deleted
    :return: a pair composed of two 1d arrays: vertex_perimeter and edge_length
    """
    vertex_map = hg.CptRegionAdjacencyGraph.get_vertex_map(rag)
    edge_map = hg.CptRegionAdjacencyGraph.get_edge_map(rag)

    if len(shape) != 2:
        raise ValueError("Graph must be a grid graph of dimension 2.")

    vertex_perimeter, edge_length = \
        hg.cpp._rag_2d_vertex_perimeter_and_edge_length(
            rag,
            vertex_map,
            edge_map,
            shape,
            pre_graph,
            epsilon,
            relative_epsilon,
            min_size)

    return vertex_perimeter, edge_length
