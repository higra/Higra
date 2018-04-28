import higra as hg


@hg.dataConsumer("vertexWeights")
def weightGraph(graph, vertexWeights, weightFunction):
    """
    Compute the edge weights of a graph using source and target vertices values
    and specified weighting function (see WeightFunction enumeration).

    Set the attribute "vertexWeights" of graph with the provided values.
    Set the attribute "edgeWeights" of graph with the computed values.

    :param graph:
    :param vertexWeights:
    :param weightFunction: in WeightFunction enumeration
    :return: edgeWeights
    """

    edgeWeights = hg._weighGraph(graph, weightFunction, vertexWeights)
    hg.setAttribute(graph, "vertexWeights", vertexWeights)
    hg.setAttribute(graph, "edgeWeights", edgeWeights)

    return edgeWeights


def weightGraphFunc(graph, weightFunction):
    """
    Compute the edge weights of a graph with the given weighting function.
    The weighting function takes the vertex index of the extremities of an edge and returns the weight of the edge (scalar)

    Set the attribute "edgeWeights" of graph with the computed values.

    :param graph:
    :param weightFunction, eg. lambda i, j: abs(data[i]-[j])
    :return: edgeWeights
    """

    edgeWeights = hg._weighGraph(graph, weightFunction)

    hg.setAttribute(graph, "edgeWeights", edgeWeights)

    return edgeWeights
