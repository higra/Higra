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


def auto_concept_docstring(cls):
    cls.__doc__ = cls.description()
    return cls


class Concept(object):
    _name = "Abstract Concept"
    _description = "A concept describes how a set of data elements can be considered as a coherent notion of higher " \
                   "semantic level. " \
                   "One of the data elements of the concept is the canonical element of the association " \
                   "that can be used to materialize the concept, ie, to assemble all the data elements associated to " \
                   "the canonical element in order to build a representation of the higher semantic object described " \
                   "by the concept.\n" \
                   "Concepts are conceptually close to Classes in object oriented programming where data elements " \
                   "correspond to attributes. " \
                   "However with concepts, one never creates or manipulates actual objects: objects exist only from " \
                   "the relation between their data elements."

    _data_elements = {}
    _canonical_data_element = "The canonical element of a concept is the data that serves as a reference to retrieve all " \
                              "the data elements of the Concept. The canonical element is usually chosen such that there " \
                              "is a natural n to 1 association between the canonical element and its data elements."

    def __init__(self, **kw_args):
        self.name_mapping = kw_args

    @staticmethod
    def link():
        raise NotImplementedError("Missing function in concept")

    @classmethod
    def validate(cls, canonical_element):
        """
        Check if the given canonical is tagged with the given Concept (class on which the `validate` function is called).

        :param canonical_element:
        :return:
        """
        if canonical_element is None:
            return False
        return hg.has_tag(canonical_element, cls)

    @classmethod
    def construct(cls, canonical_element, strict=True, data_cache=None):
        """
        Tries to construct a dictionary holding all the data elements of the concept associated to the given canonical element.

        :param canonical_element: an object
        :param strict: if `True` an exception is raised if all the data elements of the Concept cannot be recovered from the given canonical element, otherwise the missing data element is simply not included in the result.
        :param data_cache: specify a data cache where to search for data elements. If `None`, the default Higra global data cache is used.
        :return: a dictionary
        """
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
        """
        Get a textual description of the Concept

        :return: a string with all the information available on the Concept
        """
        description = "**Concept Name**: " + cls._name + "\n\n"
        description += "**Description**: " + cls._description + "\n\n"
        description += "**Canonical data element**: " + cls._canonical_data_element + "\n\n"
        description += "**Data elements**:\n\n"
        classes = inspect.getmro(cls)
        for c in classes:
            if issubclass(c, Concept):
                for n, d in c._data_elements.items():
                    if c is not cls and d[1] is None:
                        continue
                    description += "  * *" + n + "*"
                    if d[1]:
                        description += " (attribute name: '" + d[1] + "'): "
                    else:
                        description += " (canonical): "
                    description += d[0]
                    if c != cls:
                        description += " (inherited from :class:`~higra." + c.__name__ + "`)"
                    description += "\n\n"
        return description


auto_concept_docstring(Concept)


@auto_concept_docstring
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

    @staticmethod
    def get_shape(graph):
        """
        The shape (a tuple of positive integers) of the graph.

        :param graph:
        :return:
        """
        return hg.get_attribute(graph, "shape")


@auto_concept_docstring
class CptRegionAdjacencyGraph(Concept):
    _name = "region_adjacency_graph"
    _description = "A graph whose vertices represented super-vertices of another graph."
    _data_elements = {"rag": ("the region adjacency graph", None),
                      "pre_graph": ("the base graph", "pre_graph"),
                      "vertex_map": ("a map giving for each vertex of pre_graph the corresponding vertex of rag",
                                     "vertex_map"),
                      "edge_map": ("a map giving for each edge of pre_graph the corresponding edge of rag "
                                   "(and -1 if not such edge exists)", "edge_map")}
    _canonical_data_element = "rag"

    def __init__(self, **kwargs):
        super(CptRegionAdjacencyGraph, self).__init__(**kwargs)

    @staticmethod
    def link(rag, pre_graph, vertex_map, edge_map):
        hg.add_tag(rag, CptRegionAdjacencyGraph)
        hg.set_attribute(rag, "vertex_map", vertex_map)
        hg.set_attribute(rag, "edge_map", edge_map)
        hg.set_attribute(rag, "pre_graph", pre_graph)

    @staticmethod
    def get_vertex_map(rag):
        """
        The map giving for each vertex of pre_graph of the given rag, i.e. the graph the rag was built on, the corresponding vertex of rag.

        :param rag:
        :return:
        """
        return hg.get_attribute(rag, "vertex_map")

    @staticmethod
    def get_edge_map(rag):
        """
        The map giving for each edge of pre_graph of the given rag, i.e. the graph the rag was built on, the corresponding edge of rag (and -1 if not such edge exists).

        :param rag:
        :return:
        """
        return hg.get_attribute(rag, "edge_map")

    @staticmethod
    def get_pre_graph(rag):
        """
        The graph the rag was built on.

        :param rag:
        :return:
        """
        return hg.get_attribute(rag, "pre_graph")


@auto_concept_docstring
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

    @staticmethod
    def get_leaf_graph(tree):
        """
        The graph the tree was built on, i.e. the graph associated to the leaves of the tree.

        :param tree:
        :return:
        """
        return hg.get_attribute(tree, "leaf_graph")


@auto_concept_docstring
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

    @staticmethod
    def get_mst(tree):
        """
        The minimum spanning tree of the leaf graph of the hierarchy.

        :param tree:
        :return:
        """
        return hg.get_attribute(tree, "mst")


@auto_concept_docstring
class CptMinimumSpanningTree(Concept):
    _name = "minimum_spanning_tree"
    _description = "A minimum spanning tree and its base graph."
    _data_elements = {"base_graph": ("a base graph", "base_graph"),
                      "mst": ("a minimum spanning tree of the base graph", None),
                      "mst_edge_map": (
                          "For each edge index i of the mst, gives the corresponding edge index in the base graph",
                          "mst_edge_map")}
    _canonical_data_element = "mst"

    def __init__(self, **kwargs):
        super(CptMinimumSpanningTree, self).__init__(**kwargs)

    @staticmethod
    def link(mst, base_graph, mst_edge_map):
        hg.add_tag(mst, CptMinimumSpanningTree)
        hg.set_attribute(mst, "base_graph", base_graph)
        hg.set_attribute(mst, "mst_edge_map", mst_edge_map)

    @staticmethod
    def get_base_graph(mst):
        """
        The graph on which the given mst was computed.

        :param mst:
        :return:
        """
        return hg.get_attribute(mst, "base_graph")

    @staticmethod
    def get_edge_map(mst):
        """
        The map that gives for each edge index i of the mst, the corresponding edge index in the base graph

        :param mst:
        :return:
        """
        return hg.get_attribute(mst, "mst_edge_map")
