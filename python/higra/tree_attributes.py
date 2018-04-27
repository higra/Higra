import higra.data_cache as dc
import numpy as np
import higra as hg


@dc.dataProvider("leavesArea")
def attributeLeavesArea(tree):
    return np.ones((tree.numLeaves(),))


@dc.dataProvider("area")
@dc.dataConsumer("leavesArea")
def attributeArea(tree, leavesArea):
    return tree.accumulateSequential(leavesArea, hg.Accumulators.sum)


@dc.dataProvider("volume")
@dc.dataConsumer("area", "altitudes")
def attributeVolume(tree, area, altitudes):
    height = np.abs(altitudes[tree.parents()] - altitudes)
    height = height * area
    volumeLeaves = height[:tree.numLeaves()]
    return tree.accumulateAndAddSequential(height, volumeLeaves, hg.Accumulators.sum)


@dc.dataProvider("lcaMap")
@dc.dataConsumer("leavesGraph")
def attributeLCA(tree, leavesGraph):
    lca = hg.LCAFast(tree)
    return lca.lca(leavesGraph)


@dc.dataProvider("frontierLength")
@dc.dataConsumer("lcaMap")
def attributeFrontierLength(tree, lcaMap):
    frontierLength = np.zeros((tree.numVertices(),), dtype=np.int64)
    np.add.at(frontierLength, lcaMap, 1)
    return frontierLength


@dc.dataProvider("perimeterLength")
@dc.dataConsumer("leavesGraph", "frontierLength")
def attributePerimeter(tree, leavesGraph, frontierLength):
    vertices = np.arange(tree.numLeaves())
    perimeterLeaves = leavesGraph.outDegree(vertices)
    return tree.accumulateAndAddSequential(-2 * frontierLength, perimeterLeaves, hg.Accumulators.sum)


@dc.dataProvider("compactness")
@dc.dataConsumer("area", "perimeterLength")
def attributeCompactness(area, perimeterLength):
    compac = area / (perimeterLength * perimeterLength)
    return compac / np.max(compac)


@dc.dataProvider("meanData")
@dc.dataConsumer("area", "leavesData")
def attributeMeanData(tree, area, leavesData):
    return tree.accumulateSequential(leavesData.astype(np.float64), hg.Accumulators.sum) / area.reshape((-1, 1))
