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
import inspect
import functools


class Concept(object):
    _name = "abstract_concept"
    _description = "A concept describes a set of data elements whose union can be considered " \
                   "as a coherent notion of higher semantic level."
    _data_elements = {}
    _canonical_data_element = None

    def __init__(self, **kw_args):
        self.name_mapping = kw_args

    @staticmethod
    def link():
        raise NotImplementedError("Missing function in concept")

    @classmethod
    def validate(cls, canonical_element):
        if canonical_element is None:
            return False
        return hg.has_tag(canonical_element, cls)

    @classmethod
    def construct(cls, canonical_element, strict=True, data_cache=None):
        if canonical_element is None:
            return {}
        if strict and not cls.validate(canonical_element):
            raise Exception("Cannot construct concept '" + str(cls) + "': the element '" + str(canonical_element)
                            + "' does not satisfy this concept.")
        result = {cls._canonical_data_element: canonical_element}

        classes = inspect.getmro(cls)
        for c in classes:
            if issubclass(c, Concept) and c is not Concept:
                ce_name = c._canonical_data_element
                if ce_name not in result:
                    if strict:
                        raise Exception("Construction of concept '"
                                        + str(cls)
                                        + "' failed: "
                                          "cannot find canonical element named '" + ce_name + "'('" + str(c) + "')")
                else:
                    ce = result[ce_name]
                    for data_name, data_description in c._data_elements.items():
                        if data_description[1] is not None:
                            if data_cache is None:
                                elem = hg.get_attribute(ce, data_description[1])
                            else:
                                elem = data_cache.get_data(ce).get(data_description[1], None)
                            if elem is None:
                                if strict:
                                    raise Exception("Construction of concept '"
                                                    + str(cls)
                                                    + "' failed: "
                                                      "cannot find data element named '" + data_description[1] + "'('"
                                                    + str(ce) + "')")
                            else:
                                result[data_name] = elem

        return result

    @classmethod
    def description(cls):
        description = "Name: " + cls._name + "\n"
        description += "Description: " + cls._description + "\n"
        description += "Canonical data element: " + cls._canonical_data_element + "\n"
        description += "Data elements:\n"
        classes = inspect.getmro(cls)
        for c in classes:
            if issubclass(c, Concept):
                for n, d in c._data_elements.items():
                    if c is not cls and d[1] is not None:
                        continue
                    description += "\t" + n
                    if d[1]:
                        description += "('" + d[1] + "'):"
                    else:
                        description += "(canonical):"
                    description += d[0]
                    if c != cls:
                        description += " (from " + c._name + ")"
                    description += "\n"


class CptEdgeWeightedGraph(Concept):
    _name = "edge_weighted_graph"
    _description = "An edge weighted graph is the composition of a graph data structure and an array of edge weights."
    _data_elements = {"graph": ("a graph g with indexed edges", "domain"),
                      "edge_weights": ("an array a such that a.dimension >= 1 and a.shape[0] == g.num_edges()", None)}
    _canonical_data_element = "edge_weights"

    def __init__(self, **kwargs):
        super(CptEdgeWeightedGraph, self).__init__(**kwargs)

    @staticmethod
    def link(edge_weights, graph):
        hg.add_tag(edge_weights, CptEdgeWeightedGraph)
        hg.set_attribute(edge_weights, "domain", graph)


class CptGraphCut(CptEdgeWeightedGraph):
    _name = "graph_cut"
    _description = "A graph cut is the composition of a graph data structure and an array of edge weights " \
                   "such that each each edge with a non zero weight is considered to belong to the cut."

    def __init__(self, **kwargs):
        super(CptGraphCut, self).__init__(**kwargs)

    @staticmethod
    def link(edge_weights, graph):
        CptEdgeWeightedGraph.link(edge_weights, graph)
        hg.add_tag(edge_weights, CptGraphCut)
        hg.set_attribute(edge_weights, "domain", graph)


class CptVertexWeightedGraph(Concept):
    _name = "vertex_weighted_graph"
    _description = "A vertex weighted graph is the composition of a graph data structure and an array of vertex weights."
    _data_elements = {"graph": ("a graph g with indexed vertices", "domain"),
                      "vertex_weights": (
                          "an array a such that a.dimension >= 1 and a.shape[0] == g.num_vertices()", None)}
    _canonical_data_element = "vertex_weights"

    def __init__(self, **kwargs):
        super(CptVertexWeightedGraph, self).__init__(**kwargs)

    @staticmethod
    def link(vertex_weights, graph):
        hg.add_tag(vertex_weights, CptVertexWeightedGraph)
        hg.set_attribute(vertex_weights, "domain", graph)


class CptVertexLabeledGraph(CptVertexWeightedGraph):
    _name = "vertex_labeled_graph"
    _description = "A vertex labeled graph is the composition of a graph data structure and an array of vertex labels."

    def __init__(self, **kwargs):
        super(CptVertexLabeledGraph, self).__init__(**kwargs)

    @staticmethod
    def link(vertex_labels, graph):
        CptVertexWeightedGraph.link(vertex_labels, graph)
        hg.add_tag(vertex_labels, CptVertexLabeledGraph)


class CptGridGraph(Concept):
    _name = "grid_graph"
    _description = "A graph whose vertices correspond to points on a regular grid."
    _data_elements = {"graph": ("a graph g", None),
                      "shape": ("the shape of the underlying grid (a 1d array of positive integers " \
                                "such that np.prod(shape) = g.num_vertices())", "shape")}
    _canonical_data_element = "graph"

    def __init__(self, **kwargs):
        super(CptGridGraph, self).__init__(**kwargs)

    @staticmethod
    def link(graph, shape):
        hg.add_tag(graph, CptGridGraph)
        hg.set_attribute(graph, "shape", shape)


class CptRegionAdjacencyGraph(Concept):
    _name = "region_adjacency_graph"
    _description = "A graph whose vertices represented super-vertices of another graph."
    _data_elements = {"rag": ("the region adjacency graph", None),
                      "pre_graph": ("the base graph", "pre_graph"),
                      "vertex_map": ("a map giving for each vertex of pre_graph the corresponding vertex of rag",
                                     "vertex_map"),
                      "edge_map": ("a map giving for each edge of pre_graph the corresponding edge of rag "
                                   "(and -1 if not such edge exists", "edge_map")}
    _canonical_data_element = "rag"

    def __init__(self, **kwargs):
        super(CptRegionAdjacencyGraph, self).__init__(**kwargs)

    @staticmethod
    def link(rag, pre_graph, vertex_map, edge_map):
        hg.add_tag(rag, CptRegionAdjacencyGraph)
        hg.set_attribute(rag, "vertex_map", vertex_map)
        hg.set_attribute(rag, "edge_map", edge_map)
        hg.set_attribute(rag, "pre_graph", pre_graph)
        CptEdgeWeightedGraph.link(edge_map, pre_graph)
        CptVertexLabeledGraph.link(vertex_map, pre_graph)


class CptSaliencyMap(CptEdgeWeightedGraph):
    _name = "saliency_map"
    _description = "A saliency map represents a hierarchy by an edge weighted graph."

    def __init__(self, **kwargs):
        super(CptSaliencyMap, self).__init__(**kwargs)

    @staticmethod
    def link(edge_weights, graph):
        hg.add_tag(edge_weights, CptSaliencyMap)
        CptEdgeWeightedGraph.link(edge_weights, graph)


class CptHierarchy(Concept):
    _name = "hierarchy"
    _description = "A tree representing a hierarchy over a base graph."
    _data_elements = {"tree": ("a tree t", None),
                      "leaf_graph": ("the graph associated to the leaves of the tree", "leaf_graph"),
                      }
    _canonical_data_element = "tree"

    def __init__(self, **kwargs):
        super(CptHierarchy, self).__init__(**kwargs)

    @staticmethod
    def link(tree, leaf_graph):
        hg.add_tag(tree, CptHierarchy)
        hg.set_attribute(tree, "leaf_graph", leaf_graph)


class CptValuedHierarchy(CptHierarchy):
    _name = "valued_hierarchy"
    _description = "A tree representing a hierarchy with node values."
    _data_elements = {"tree": ("a hierarchy", "domain"),
                      "altitudes": ("an array a such that a.dimension >= 1 and a.shape[0] == t.num_vertices()", None)}
    _canonical_data_element = "altitudes"

    def __init__(self, **kwargs):
        super(CptValuedHierarchy, self).__init__(**kwargs)

    @staticmethod
    def link(altitudes, hierarchy):
        hg.add_tag(altitudes, CptValuedHierarchy)
        hg.set_attribute(altitudes, "domain", hierarchy)


class CptBinaryHierarchy(CptHierarchy):
    _name = "binary_hierarchy"
    _description = "A binary tree representing a hierarchy with an associated " \
                   "minimum spanning tree on the hierarchy leaf graph."
    _data_elements = {"tree": ("a hierarchy h", None),
                      "mst": ("a minimum spanning tree of the leaf graph of the hierarchy", "mst")}
    _canonical_data_element = "tree"

    def __init__(self, **kwargs):
        super(CptBinaryHierarchy, self).__init__(**kwargs)

    @staticmethod
    def link(hierarchy, mst):
        hg.add_tag(hierarchy, CptBinaryHierarchy)
        hg.set_attribute(hierarchy, "mst", mst)


class CptMinimumSpanningTree(Concept):
    _name = "minimum_spanning_tree"
    _description = "A minimum spanning tree and its base graph."
    _data_elements = {"base_graph": ("a base graph", "base_graph"),
                      "mst": ("a minimum spanning tree of the base graph", None),
                      "mst_edge_map": ("For each edge index i of the mst, gives the corresponding edge index in the base graph", "mst_edge_map")}
    _canonical_data_element = "mst"

    def __init__(self, **kwargs):
        super(CptMinimumSpanningTree, self).__init__(**kwargs)

    @staticmethod
    def link(mst, base_graph, mst_edge_map):
        hg.add_tag(mst, CptMinimumSpanningTree)
        hg.set_attribute(mst, "base_graph", base_graph)
        hg.set_attribute(mst, "mst_edge_map", mst_edge_map)

