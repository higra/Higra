from higra.attributes_util import HGAttribute
import numpy as np
import higra as hg


@HGAttribute("leavesArea")
def attributeLeavesArea(tree):
    return np.ones((tree.numLeaves(),))


@HGAttribute("area", ("leavesArea",))
def attributeArea(tree, leavesArea):
    return tree.accumulateSequential(leavesArea, hg.Accumulators.sum)


@HGAttribute("volume", ("area", "altitudes"))
def attributeVolume(tree, area, altitudes):
    height = np.abs(altitudes[tree.parents()] - altitudes)
    height = height * area
    volumeLeaves = height[:tree.numLeaves()]
    return tree.accumulateAndAddSequential(height, volumeLeaves, hg.Accumulators.sum)


@HGAttribute("lcaMap", ("leavesGraph",))
def attributeLCA(tree, leavesGraph):
    lca = hg.LCAFast(tree)
    return lca.lca(leavesGraph)


@HGAttribute("frontierLength", ("lcaMap",))
def attributeFrontierLength(tree, lcaMap):
    frontierLength = np.zeros((tree.numVertices(),), dtype=np.int64)
    np.add.at(frontierLength, lcaMap, 1)
    return frontierLength


@HGAttribute("perimeterLength", ("leavesGraph", "frontierLength"))
def attributePerimeter(tree, leavesGraph, frontierLength):
    vertices = np.arange(tree.numLeaves())
    perimeterLeaves = leavesGraph.outDegree(vertices)
    return tree.accumulateAndAddSequential(-2 * frontierLength, perimeterLeaves, hg.Accumulators.sum)


@HGAttribute("compactness", ("area", "perimeterLength"))
def attributeCompactness(area, perimeterLength):
    compac = area / (perimeterLength * perimeterLength)
    return compac / np.max(compac)


@HGAttribute("meanData", ("area", "leavesData"))
def attributeMeanData(tree, area, leavesData):
    return tree.accumulateSequential(leavesData.astype(np.float64), hg.Accumulators.sum) / area.reshape((-1, 1))
