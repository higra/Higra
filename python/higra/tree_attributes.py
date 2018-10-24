############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################


import numpy as np
import higra as hg


@hg.data_provider("vertex_area")
def attribute_vertex_area(graph):
    pre_graph = hg.get_attribute(graph, "pre_graph")
    if pre_graph: # this is a rag like graph
        pre_graph_vertex_area = attribute_vertex_area(pre_graph)
        return hg.rag_accumulate_on_vertices(graph, hg.Accumulators.sum, vertex_weights=pre_graph_vertex_area)
    return np.ones((graph.num_vertices(),), dtype=np.int64)


@hg.data_provider("vertex_perimeter")
def attribute_vertex_perimeter(graph):
    vertices = np.arange(graph.num_vertices(), dtype=np.int64)
    return graph.out_degree(vertices)


@hg.data_provider("area")
@hg.data_consumer(leaf_area="leaf_graph.vertex_area")
def attribute_area(tree, leaf_area):
    return hg.accumulate_sequential(tree, leaf_area, hg.Accumulators.sum)


@hg.data_provider("volume")
@hg.data_consumer("area", "altitudes")
def attribute_volume(tree, area, altitudes):
    height = np.abs(altitudes[tree.parents()] - altitudes)
    height = height * area
    volume_leaves = height[:tree.num_leaves()]
    return hg.accumulate_and_add_sequential(tree, height, volume_leaves, hg.Accumulators.sum)


@hg.data_provider("lca_map")
@hg.data_consumer("leaf_graph")
def attribute_lca_map(tree, leaf_graph):
    lca = hg.LCAFast(tree)
    return lca.lca(leaf_graph)


@hg.data_provider("frontier_length")
@hg.data_consumer("lca_map")
def attribute_frontier_length(tree, lca_map):
    frontier_length = np.zeros((tree.num_vertices(),), dtype=np.int64)
    np.add.at(frontier_length, lca_map, 1)
    return frontier_length


@hg.data_provider("perimeter_length")
@hg.data_consumer("frontier_length", leaf_perimeter="leaf_graph.vertex_perimeter")
def attribute_perimeter_length(tree, leaf_perimeter, frontier_length):
    return hg.accumulate_and_add_sequential(tree, -2 * frontier_length, leaf_perimeter, hg.Accumulators.sum)


@hg.data_provider("compactness")
@hg.data_consumer("area", "perimeter_length")
def attribute_compactness(tree, area, perimeter_length):
    compactness = area / (perimeter_length * perimeter_length)
    max_compactness = np.nanmax(compactness)
    return compactness / max_compactness


@hg.data_provider("mean_weights")
@hg.data_consumer("area", leaf_data="leaf_graph.vertex_weights")
def attribute_mean_weights(tree, area, leaf_data):
    return hg.accumulate_sequential(tree, leaf_data.astype(np.float64), hg.Accumulators.sum) / area.reshape((-1, 1))
