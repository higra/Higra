
import numpy as np
import higra as hg


@hg.dataProvider("vertexArea")
def attributeVertexArea(graph):
    return np.ones((graph.numVertices(),))


@hg.dataProvider("area")
@hg.dataConsumer(leafArea="leafGraph.vertexArea")
def attributeArea(tree, leafArea):
    return tree.accumulateSequential(leafArea, hg.Accumulators.sum)


@hg.dataProvider("volume")
@hg.dataConsumer("area", "altitudes")
def attributeVolume(tree, area, altitudes):
    height = np.abs(altitudes[tree.parents()] - altitudes)
    height = height * area
    volumeLeaves = height[:tree.numLeaves()]
    return tree.accumulateAndAddSequential(height, volumeLeaves, hg.Accumulators.sum)


@hg.dataProvider("lcaMap")
@hg.dataConsumer("leafGraph")
def attributeLCA(tree, leafGraph):
    lca = hg.LCAFast(tree)
    return lca.lca(leafGraph)


@hg.dataProvider("frontierLength")
@hg.dataConsumer("lcaMap")
def attributeFrontierLength(tree, lcaMap):
    frontierLength = np.zeros((tree.numVertices(),), dtype=np.int64)
    np.add.at(frontierLength, lcaMap, 1)
    return frontierLength


@hg.dataProvider("perimeterLength")
@hg.dataConsumer("leafGraph", "frontierLength")
def attributePerimeter(tree, leafGraph, frontierLength):
    vertices = np.arange(tree.numLeaves())
    perimeterLeaves = leafGraph.outDegree(vertices)
    return tree.accumulateAndAddSequential(-2 * frontierLength, perimeterLeaves, hg.Accumulators.sum)


@hg.dataProvider("compactness")
@hg.dataConsumer("area", "perimeterLength")
def attributeCompactness(area, perimeterLength):
    compac = area / (perimeterLength * perimeterLength)
    return compac / np.max(compac)


@hg.dataProvider("meanWeights")
@hg.dataConsumer("area", leafData="leafGraph.vertexWeights")
def attributeMeanWeights(tree, area, leafData):
    return tree.accumulateSequential(leafData.astype(np.float64), hg.Accumulators.sum) / area.reshape((-1, 1))
