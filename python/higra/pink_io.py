import higra as hg


def read_graph_pink(filename):
    """
    Read a graph file stored in pink ascii format
    :param filename:
    :return: Graph (with attributes "vertex_weights", "edge_weights", "shape")
    """
    graph, vertexWeights, edgeWeights, shape = hg._read_graph_pink(filename)

    hg.set_attribute(graph, "edge_weigths", edgeWeights)
    hg.set_attribute(graph, "vertex_weigths", vertexWeights)
    hg.set_attribute(graph, "shape", shape)

    return graph
