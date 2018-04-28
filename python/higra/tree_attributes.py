import higra.data_cache as dc
import numpy as np
import higra as hg


@dc.dataProvider("vertexArea")
def attributeVertexArea(graph):
    return np.ones((graph.numLeaves(),))


@dc.dataProvider("area")
@dc.dataConsumer(leafArea="leafGraph.vertexArea")
def attributeArea(tree, leafArea):
    return tree.accumulateSequential(leafArea, hg.Accumulators.sum)


@dc.dataProvider("volume")
@dc.dataConsumer("area", "altitudes")
def attributeVolume(tree, area, altitudes):
    height = np.abs(altitudes[tree.parents()] - altitudes)
    height = height * area
    volumeLeaves = height[:tree.numLeaves()]
    return tree.accumulateAndAddSequential(height, volumeLeaves, hg.Accumulators.sum)


@dc.dataProvider("lcaMap")
@dc.dataConsumer("leafGraph")
def attributeLCA(tree, leafGraph):
    lca = hg.LCAFast(tree)
    return lca.lca(leafGraph)


@dc.dataProvider("frontierLength")
@dc.dataConsumer("lcaMap")
def attributeFrontierLength(tree, lcaMap):
    frontierLength = np.zeros((tree.numVertices(),), dtype=np.int64)
    np.add.at(frontierLength, lcaMap, 1)
    return frontierLength


@dc.dataProvider("perimeterLength")
@dc.dataConsumer("leafGraph", "frontierLength")
def attributePerimeter(tree, leafGraph, frontierLength):
    vertices = np.arange(tree.numLeaves())
    perimeterLeaves = leafGraph.outDegree(vertices)
    return tree.accumulateAndAddSequential(-2 * frontierLength, perimeterLeaves, hg.Accumulators.sum)


@dc.dataProvider("compactness")
@dc.dataConsumer("area", "perimeterLength")
def attributeCompactness(area, perimeterLength):
    compac = area / (perimeterLength * perimeterLength)
    return compac / np.max(compac)


@dc.dataProvider("meanWeights")
@dc.dataConsumer("area", leafData="leafGraph.vertexWeights")
def attributeMeanWeights(tree, area, leafData):
    return tree.accumulateSequential(leafData.astype(np.float64), hg.Accumulators.sum) / area.reshape((-1, 1))
