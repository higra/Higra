import higra as hg


@hg.dataConsumer("edgeWeights")
def bptCanonical(graph, edgeWeights):
    """
    Compute the canonical binary partition tree (binary tree by altitude ordering) of the given weighted graph.


    :param graph:
    :param edgeWeights:
    :return: Tree (with attributes "leafGraph", "altitudes" and "mst")
    """

    tree, altitudes, mst = hg._bptCanonical(graph, edgeWeights)

    originalGraph = hg.getAttribute(graph, "originalGraph")
    if originalGraph:
        hg.setAttribute(tree, "leafGraph", originalGraph)
        hg.setAttribute(mst, "originalGraph", originalGraph)
    else:
        hg.setAttribute(tree, "leafGraph", graph)
        hg.setAttribute(mst, "originalGraph", graph)

    hg.setAttribute(tree, "altitudes", altitudes)
    hg.setAttribute(tree, "mst", mst)


    return tree


@hg.dataConsumer("deletedVertices")
def simplifyTree(tree, deletedVertices):
    """
    Creates a copy of the current Tree and deletes the vertices i such that deletedVertices[i] is true.

    The attribute "nodeMap" of the returned tree is an array that maps any node index i of the new tree,
    to the index of this node in the original tree.

    :param tree:
    :param deletedVertices:
    :return: simplified tree (with attributes "nodeMap")
    """

    ntree, nodeMap = hg._simplifyTree(tree, deletedVertices)

    hg.setAttribute(ntree, "leafGraph", hg.getAttribute(tree, "leafGraph"))
    hg.setAttribute(ntree, "nodeMap", nodeMap)

    return ntree


@hg.dataConsumer("altitudes", "lcaMap")
def saliency(tree, altitudes, lcaMap):
    """
    Compute the saliency map of the given tree
    :param tree:
    :param altitudes: altitudes of the vertices of the tree
    :param lcaMap: array containing the lowest common ancestor of the source and target vertices of each edge
    where saliency need to be computed
    :return: altitudes[lcaMap]
    """

    return altitudes[lcaMap]
