import higra as hg


@hg.dataConsumer("shape", "edgeWeights")
def contour2Khalimsky(graph, shape, edgeWeights, addExtraBorder=False):
    """
    Create a contour image in the Khalimsky grid from a 4 adjacency edge-weighted graph.

    :param graph: must be a 4 adjacency 2d graph
    :param shape: shape of the graph
    :param edgeWeights: edge weights of the graph
    :param addExtraBorder: if False result size is 2 * shape - 1 and 2 * shape +1 otherwise
    :return:
    """

    return hg._contour2Khalimsky(graph, shape, edgeWeights, addExtraBorder)


def khalimsky2Contour(khalimsky, extraBorder=False):
    """
    Create a 4 adjacency edge-weighted graph from a contour image in the Khalimsky grid.

    :param khalimsky:
    :param extraBorder:
    :return: Graph (with attributes "edgeWeights" and "shape")
    """

    graph, embedding, edgeWeights = hg._khalimsky2Contour(khalimsky, extraBorder)

    hg.setAttribute(graph, "shape", embedding.shape())
    hg.setAttribute(graph, "edgeWeights", edgeWeights)

    return graph


def get4AdjacencyGraph(shape):
    """
    Create an explicit undirected 4 adjacency graph of the given shape.
    :param shape:
    :return: Graph (with attribute "shape")
    """
    graph = hg._get4AdjacencyGraph(shape)
    hg.setAttribute(graph, "shape", shape)
    return graph


def get8AdjacencyGraph(shape):
    """
    Create an explicit undirected 8 adjacency graph of the given shape.
    :param shape:
    :return: Graph (with attribute "shape")
    """
    graph = hg._get8AdjacencyGraph(shape)
    hg.setAttribute(graph, "shape", shape)
    return graph


def get4AdjacencyImplicitGraph(shape):
    """
    Create an implicit undirected 4 adjacency graph of the given shape (edges are not stored).
    :param shape:
    :return: Graph (with attribute "shape")
    """
    graph = hg._get4AdjacencyImplicitGraph(shape)
    hg.setAttribute(graph, "shape", shape)
    return graph


def get8AdjacencyImplicitGraph(shape):
    """
    Create an implicit undirected 8 adjacency graph of the given shape (edges are not stored).
    :param shape:
    :return: Graph (with attribute "shape")
    """
    graph = hg._get8AdjacencyImplicitGraph(shape)
    hg.setAttribute(graph, "shape", shape)
    return graph
