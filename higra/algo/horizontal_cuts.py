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


@hg.extend_class(hg.HorizontalCutNodes, method_name="reconstruct_leaf_data")
@hg.argument_helper(("altitudes", hg.CptValuedHierarchy))
def __reconstruct_leaf_data(self, altitudes, tree):
    leaf_weights = self._reconstruct_leaf_data(tree, altitudes)

    if hg.CptHierarchy.validate(tree):
        leaf_graph = hg.CptHierarchy.construct(tree)["leaf_graph"]
        leaf_weights = hg.delinearize_vertex_weights(leaf_weights, leaf_graph)
        hg.CptVertexWeightedGraph.link(leaf_weights, leaf_graph)

    return leaf_weights


@hg.extend_class(hg.HorizontalCutNodes, method_name="graph_cut")
@hg.argument_helper(("tree", hg.CptHierarchy))
def __graph_cut(self, tree, leaf_graph, handle_rag=True):
    sm = self._graph_cut(tree, leaf_graph)

    if hg.CptRegionAdjacencyGraph.validate(leaf_graph) and handle_rag:
        sm = hg.rag_back_project_edge_weights(sm, leaf_graph)
        hg.CptSaliencyMap.link(sm, hg.get_attribute(leaf_graph, "pre_graph"))
    else:
        hg.CptSaliencyMap.link(sm, leaf_graph)

    return sm


@hg.extend_class(hg.HorizontalCutExplorer, method_name="__new__")
@hg.argument_helper(("altitudes", hg.CptValuedHierarchy))
def __make_HorizontalCutExplorer(cls, altitudes, tree):
    return cls._make_HorizontalCutExplorer(tree, altitudes)


@hg.extend_class(hg.HorizontalCutExplorer, method_name="__init__")
def __dummy_init_HorizontalCutExplorer(*_):
    pass
