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
        return hg.has_tag(canonical_element, cls)

    @classmethod
    def construct(cls, canonical_element, strict=True):
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
                            elem = hg.get_attribute(ce, data_description[1])
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
        hg.add_attribute(edge_weights, "domain", graph)


class CptVertexWeightedGraph(Concept):
    _name = "vertex_weighted_graph"
    _description = "A vertex weighted graph is the composition of a graph data structure and an array of vertex weights."
    _data_elements = {"graph": ("a graph g with indexed vertices", "domain"),
                      "edge_weights": (
                      "an array a such that a.dimension >= 1 and a.shape[0] == g.num_vertices()", None)}
    _canonical_data_element = "vertex_weights"

    def __init__(self, **kwargs):
        super(CptVertexWeightedGraph, self).__init__(**kwargs)

    @staticmethod
    def link(vertex_weights, graph):
        hg.add_tag(vertex_weights, CptVertexWeightedGraph)
        hg.add_attribute(vertex_weights, "domain", graph)


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
        hg.add_attribute(graph, "shape", shape)


class CptRegionAdjacencyGraph(Concept):
    _name = "region_adjacency_graph"
    _description = "A graph whose vertices represented super-vertices of another graph."
    _data_elements = {"rag": ("the region adjacency graph", None),
                      "pre_graph": ("the base graph", "pre_graph"),
                      "vertex_map": ("a map giving for each vertex of pre_graph the corresponding vertex of rag",
                                     "vertex_map"),
                      "edge_map": ("a map giving for each edge of pre_graph the corresponding edge of rag "
                                   "(and -1 if not such edge exists", "vertex_map")}
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
        CptVertexWeightedGraph.link(vertex_map, pre_graph)


class CpSaliencyMap(CptEdgeWeightedGraph):
    _name = "saliency_map"
    _description = "A saliency map represents a hierarchy by an edge weighted graph."

    def __init__(self, **kwargs):
        super(CpSaliencyMap, self).__init__(**kwargs)

    @staticmethod
    def link(edge_weights, graph):
        hg.add_tag(edge_weights, CpSaliencyMap)
        CptEdgeWeightedGraph.link(edge_weights, graph)


class CptDendrogram(Concept):
    _name = "dendrogram"
    _description = "A vertex weighted tree representing a hierarchy."
    _data_elements = {"tree": ("a tree t", "domain"),
                      "altitudes": ("an array a such that a.dimension >= 1 and a.shape[0] == t.num_vertices()", None)}
    _canonical_data_element = "altitudes"

    def __init__(self, **kwargs):
        super(CptDendrogram, self).__init__(**kwargs)

    @staticmethod
    def link(altitudes, tree):
        hg.add_tag(altitudes, CptDendrogram)
        hg.add_attribute(altitudes, "domain", tree)


def __transfer_to_kw_arguments(signature, args, kwargs):
    nargs = list(args)
    for p in signature.parameters.values():
        if len(nargs) == 0:
            break
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            kwargs[p.name] = nargs[0]
            del nargs[0]
    return nargs


def data_concepts(*concepts):
    def decorator(fun):

        signature = inspect.signature(fun)
        all_parameters_name = set()
        for p in signature.parameters.values():
            all_parameters_name.add(p.name)

        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            args = __transfer_to_kw_arguments(signature, args, kwargs)
            data_debug = kwargs.pop("data_debug", False)
            data_cache = kwargs.pop("data_cache", hg.__higra_global_cache)
            all_data_found = dict(kwargs)
            for concept_elem in concepts:

                try:
                    arg_name, concept = concept_elem
                except:  # failed to unpack, use first parameter name
                    concept = concept_elem
                    arg_name = signature.parameters.values().__iter__().__next__().name

                if not type(concept) is type:
                    concept_type = type(concept)
                    concept_name_to_arg_name_map = concept.name_mapping
                else:
                    concept_type = concept
                    concept_name_to_arg_name_map = {}

                if not issubclass(concept_type, Concept):
                    raise Exception(str(concept_type) + " is not a subclass of the abstract Concept class.")
                if arg_name in all_data_found:
                    arg_value = all_data_found[arg_name]
                    concept_elements = concept_type.construct(arg_value, strict=False)

                    for data_element_name, data_element in concept_elements.items():
                        argument_name = concept_name_to_arg_name_map.get(data_element_name, data_element_name)
                        if argument_name not in kwargs and argument_name in all_parameters_name:
                            kwargs[argument_name] = data_element

                    all_data_found.update(concept_elements)

            return fun(*args, **kwargs)

        wrapper.original = fun

        return wrapper

    return decorator

# def concept(*args, **kwargs):
#     pass
#
# @concept(edge_weights=CpEdgeWeightedGraph)
# def bpt_canonical(edge_weights, graph):
#     pass
#
# @concept(edge_weights=CpEdgeWeightedGraph, graph=CpGridGraph)
# def graph_4_adjacency_2_khalimsky(graph, shape, edge_weights, add_extra_border=False):
#     pass
#
# @hg.data_consumer(rag=CpRegionAdjacencyGraph)
# def rag_accumulate_on_vertices(rag, accumulator, vertex_weights):
#     pass
