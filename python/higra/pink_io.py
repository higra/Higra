import higra as hg


def readGraphPink(filename):
    """
    Read a graph file stored in pink ascii format
    :param filename:
    :return: Graph (with attributes "vertexWeights", "edgeWeights", "shape")
    """
    graph, vertexWeights, edgeWeights, shape = hg._readGraphPink(filename)

    hg.setAttribute(graph, "edgeWeigths", edgeWeights)
    hg.setAttribute(graph, "vertexWeigths", vertexWeights)
    hg.setAttribute(graph, "shape", shape)

    return graph
