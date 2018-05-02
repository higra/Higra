
import numpy as np
import higra as hg


@hg.data_provider("vertex_area")
def attribute_vertex_area(graph):
    return np.ones((graph.num_vertices(),))


@hg.data_provider("area")
@hg.data_consumer(leaf_area="leaf_graph.vertex_area")
def attribute_area(tree, leaf_area):
    return tree.accumulate_sequential(leaf_area, hg.Accumulators.sum)


@hg.data_provider("volume")
@hg.data_consumer("area", "altitudes")
def attribute_volume(tree, area, altitudes):
    height = np.abs(altitudes[tree.parents()] - altitudes)
    height = height * area
    volume_leaves = height[:tree.numLeaves()]
    return tree.accumulate_and_add_sequential(height, volume_leaves, hg.Accumulators.sum)


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
@hg.data_consumer("leaf_graph", "frontier_length")
def attribute_perimeter(tree, leaf_graph, frontier_length):
    vertices = np.arange(tree.numLeaves())
    perimeter_leaves = leaf_graph.out_degree(vertices)
    return tree.accumulate_and_add_sequential(-2 * frontier_length, perimeter_leaves, hg.Accumulators.sum)


@hg.data_provider("compactness")
@hg.data_consumer("area", "perimeter_length")
def attribute_compactness(area, perimeter_length):
    compac = area / (perimeter_length * perimeter_length)
    return compac / np.max(compac)


@hg.data_provider("mean_weights")
@hg.data_consumer("area", leaf_data="leaf_graph.vertex_weights")
def attribute_mean_weights(tree, area, leaf_data):
    return tree.accumulate_sequential(leaf_data.astype(np.float64), hg.Accumulators.sum) / area.reshape((-1, 1))
