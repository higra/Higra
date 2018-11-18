############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import higra as hg


@hg.argument_helper(hg.CptHierarchy, ("leaf_graph", hg.CptRegionAdjacencyGraph))
def make_assesser_fragmentation_optimal_cut(tree,
                                            ground_truth,
                                            measure,
                                            max_regions=200,
                                            vertex_map=None):
    """
    Create an assesser for hierarchy optimal cuts w.r.t. a given ground-truth partition of hierarchy
    leaves and the given optimal cut measure (see OptimalCutMeasure). The algorithms will explore optimal
    cuts containing at most max_regions regions.

    :param tree: input hierarchy
    :param ground_truth: labelisation of base graph vertices (see parameter vertex_map if base_graph is a rag)
    :param measure: evaluation measure to use (see enumeration OptimalCutMeasure)
    :param max_regions: maximum number of regions in the cuts
    :param vertex_map: vertex mapping if the hierarchy is build on super-vertices
    :return: an object of type AssesserFragmentationOptimalCut
    """
    if vertex_map is None:
        return hg.AssesserFragmentationOptimalCut(tree, ground_truth, measure, max_regions=max_regions)
    else:
        return hg.AssesserFragmentationOptimalCut(tree, ground_truth, measure, max_regions=max_regions,
                                                  vertex_map=vertex_map)


@hg.argument_helper(hg.CptHierarchy, ("leaf_graph", hg.CptRegionAdjacencyGraph))
def assess_fragmentation_optimal_cut(tree,
                                     ground_truth,
                                     measure,
                                     max_regions=200,
                                     vertex_map=None):
    """
    Compute the fragmentation curve of the optimal cuts in a hierarchy w.r.t. a given measure.

    :param tree: input hierarchy
    :param ground_truth: labelisation of base graph vertices (see parameter vertex_map if base_graph is a rag)
    :param measure: evaluation measure to use (see enumeration OptimalCutMeasure)
    :param max_regions: maximum number of regions in the cuts
    :param vertex_map: vertex mapping if the hierarchy is build on super-vertices
    :return: an object of type FragmentationCurve
    """
    assesser = make_assesser_fragmentation_optimal_cut(tree, ground_truth, measure, max_regions, vertex_map)
    return assesser.fragmentation_curve()


@hg.argument_helper(hg.CptValuedHierarchy, ("leaf_graph", hg.CptRegionAdjacencyGraph))
def assess_fragmentation_horizontal_cut(altitudes,
                                        ground_truth,
                                        measure,
                                        tree,
                                        max_regions=200,
                                        vertex_map=None):
    """
    Compute the fragmentation curve of the horizontal cuts in a hierarchy w.r.t. a given measure.

    :param altitudes: altitudes of the nodes of the input hierarchy
    :param tree: input hierarchy
    :param ground_truth: labelisation of base graph vertices (see parameter vertex_map if base_graph is a rag)
    :param measure: evaluation measure to use (see enumeration OptimalCutMeasure)
    :param max_regions: maximum number of regions in the cuts
    :param vertex_map: vertex mapping if the hierarchy is build on super-vertices
    :return: an object of type FragmentationCurve
    """
    if vertex_map is None:
        return hg._assess_fragmentation_horizontal_cut(tree, altitudes, ground_truth, measure, max_regions=max_regions)
    else:
        return hg._assess_fragmentation_horizontal_cut(tree, altitudes, ground_truth, measure, max_regions=max_regions,
                                                       vertex_map=vertex_map)
