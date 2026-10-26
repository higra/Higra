############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import unittest
import higra as hg


class Dummy(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self.name)


class CptA(hg.Concept):
    _name = "CA"
    _description = "Concept A."
    _data_elements = {"caA": ("canonical element", None),
                      "ea1": ("element ca 1", "ea1")}
    _canonical_data_element = "caA"

    def __init__(self, **kwargs):
        super(CptA, self).__init__(**kwargs)

    @staticmethod
    def link(caA, ea1):
        CptA.tag(caA)
        caA.ea1 = ea1


class CptB(CptA):
    _name = "CB"
    _description = "Concept B."
    _data_elements = {"caB": ("canonical element", None),
                      "caA": ("element cb 1", "caA"),
                      "eb2": ("element cb 2", "eb2")}
    _canonical_data_element = "caB"

    def __init__(self, **kwargs):
        super(CptB, self).__init__(**kwargs)

    @staticmethod
    def link(caB, caA, eb2):
        CptB.tag(caB)
        caB.caA = caA
        caB.eb2 = eb2


class CptC(hg.Concept):
    _name = "CC"
    _description = "Concept C."
    _data_elements = {"caC": ("canonical element", None),
                      "ec1": ("element c 1", "ec1")}
    _canonical_data_element = "caC"

    def __init__(self, **kwargs):
        super(CptA, self).__init__(**kwargs)

    @staticmethod
    def link(caC, ec1):
        CptC.tag(caC)
        caC.ec1 = ec1


# simple implicit arg name
@hg.argument_helper(CptA)
def concept_consumer1(obj, ea1):
    return ea1


#  renamed argument
@hg.argument_helper(CptA(ea1="new_name"))
def concept_consumer2(obj, new_name):
    return new_name


# missing concepts elements in arguments
@hg.argument_helper(CptB)
def concept_consumer3(obj, ea1):
    return ea1

# chained concepts
@hg.argument_helper(CptA, ("ea1", CptC))
def concept_consumer4(obj, ec1):
    return ec1


class TestConcept(unittest.TestCase):

    def test_direct_fields_missing_and_none(self):
        graph = hg.get_4_adjacency_implicit_graph((2, 3))
        self.assertEqual(hg.CptGridGraph.construct(graph)["shape"], (2, 3))
        graph.shape = (3, 2)
        self.assertEqual(hg.CptGridGraph.get_shape(graph), (3, 2))
        del graph.shape
        self.assertTrue(callable(graph.shape))
        self.assertIsNone(hg.CptGridGraph.get_shape(graph))
        with self.assertRaisesRegex(Exception, "cannot find data element named 'shape'"):
            hg.CptGridGraph.construct(graph)
        self.assertNotIn("shape", hg.CptGridGraph.construct(graph, strict=False))
        graph.shape = None
        with self.assertRaisesRegex(Exception, "cannot find data element named 'shape'"):
            hg.CptGridGraph.construct(graph)

    def test_custom_hook_and_authoritative_fields(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            a, b, c = Dummy(1), Dummy(2), Dummy(3)
            self.assertFalse(CptA.validate(a))
            self.assertEqual(vars(a), {"name": 1})
            CptA.link(a, b)
            CptB.link(c, a, 42)
            self.assertTrue(CptB.validate(c))
            self.assertFalse(CptA.validate(c))
            self.assertEqual(concept_consumer3(c), b)
            a.ea1 = 7
            self.assertEqual(concept_consumer3(c), 7)
            self.assertEqual(concept_consumer2(a, new_name=None), 7)
            self.assertEqual(concept_consumer2(a, new_name=8), 8)
            a.ea1 = None
            self.assertNotIn("ea1", CptB.construct(c, strict=False))
            with self.assertRaisesRegex(Exception, "cannot find data element"):
                CptB.construct(c)
            del a.ea1
            with self.assertRaisesRegex(Exception, "cannot find data element"):
                CptA.construct(a)
            self.assertEqual(CptA.construct(None), {})

    def test_builtin_concept_chaining_and_maps(self):
        import numpy as np
        graph = hg.get_4_adjacency_graph((2, 3))
        rag = hg.UndirectedGraph(2)
        vertex_map = np.array([0, 0, 0, 1, 1, 1])
        edge_map = np.zeros(graph.num_edges(), dtype=np.int64)
        hg.CptRegionAdjacencyGraph.link(rag, graph, vertex_map, edge_map)
        tree = hg.Tree((2, 2, 2))
        hg.CptHierarchy.link(tree, rag)

        @hg.argument_helper(hg.CptHierarchy, ("leaf_graph", hg.CptRegionAdjacencyGraph),
                            ("pre_graph", hg.CptGridGraph))
        def consume(tree, shape=None, vertex_map=None):
            return shape, vertex_map

        shape, found_map = consume(tree)
        self.assertEqual(shape, (2, 3))
        self.assertIs(found_map, vertex_map)
        replacement = vertex_map.copy()
        rag.vertex_map = replacement
        self.assertIs(hg.CptRegionAdjacencyGraph.get_vertex_map(rag), replacement)
        self.assertIs(consume(tree)[1], replacement)
        mst = hg.UndirectedGraph(2)
        mst.add_edge(0, 1)
        mst_edge_map = np.array([0], dtype=np.int64)
        hg.CptMinimumSpanningTree.link(mst, rag, mst_edge_map)
        hg.CptBinaryHierarchy.link(tree, mst_edge_map, mst)
        for concept, owner, field in ((hg.CptHierarchy, tree, "leaf_graph"),
                                      (hg.CptRegionAdjacencyGraph, rag, "edge_map"),
                                      (hg.CptMinimumSpanningTree, mst, "base_graph"),
                                      (hg.CptBinaryHierarchy, tree, "mst_edge_map")):
            value = getattr(owner, field)
            setattr(owner, field, None)
            self.assertNotIn(field, concept.construct(owner, strict=False))
            with self.assertRaisesRegex(Exception, "cannot find data element"):
                concept.construct(owner)
            delattr(owner, field)
            with self.assertRaisesRegex(Exception, "cannot find data element"):
                concept.construct(owner)
            setattr(owner, field, value)

    def test_binary_hierarchy_mst_after_input_deletion(self):
        import gc
        import numpy as np

        for eager in (False, True):
            with self.subTest(eager=eager):
                graph = hg.UndirectedGraph(3)
                graph.add_edges([0, 0, 1], [1, 2, 2])
                input_mst = hg.minimum_spanning_tree(graph, np.array([10, 1, 2]))
                tree = hg.bpt_canonical(input_mst, np.array([1, 2]),
                                        return_altitudes=False, compute_mst=eager)
                del graph, input_mst
                gc.collect()
                concept = hg.CptBinaryHierarchy.construct(tree, strict=False)
                np.testing.assert_array_equal(concept["mst_edge_map"], [1, 2])
                mst = hg.CptBinaryHierarchy.get_mst(tree)
                self.assertIs(hg.CptBinaryHierarchy.get_mst(tree), mst)
                self.assertIs(hg.CptBinaryHierarchy.construct(tree)["mst"], mst)
                self.assertIs(mst.base_graph, concept["leaf_graph"])
                np.testing.assert_array_equal(mst.mst_edge_map, [1, 2])
                self.assertEqual(list(zip(*mst.edge_list())), [(0, 2), (1, 2)])
                self.assertEqual(list(zip(*mst.base_graph.edge_list())), [(0, 1), (0, 2), (1, 2)])

    def test_construct(self):
        o1 = Dummy(1)
        o2 = Dummy(2)
        o3 = Dummy(3)
        o4 = Dummy(4)

        CptA.link(o1, o2)
        CptB.link(o3, o1, o4)

        ca = CptA.construct(o1)
        cb = CptB.construct(o3)

        ref_ca = {"caA": o1, "ea1": o2}
        ref_cb = {"caB": o3, "eb2": o4, "caA": o1, "ea1": o2}

        self.assertTrue(ca == ref_ca)
        self.assertTrue(cb == ref_cb)

    def test_concept_consumer1(self):
        o1 = Dummy(1)
        o2 = Dummy(2)
        o3 = Dummy(3)
        o4 = Dummy(4)
        CptA.link(o1, o2)

        # call from well defined concept
        self.assertTrue(concept_consumer1(o1) == o2)

        # call from well defined concept
        self.assertTrue(concept_consumer1(obj=o1) == o2)

        # call from well defined concept with argument replacement
        self.assertTrue(concept_consumer1(o1, o3) == o3)

        # call from well defined concept with named argument replacement
        self.assertTrue(concept_consumer1(o1, ea1=o3) == o3)

        # call from well defined concept with named argument replacement
        self.assertTrue(concept_consumer1(obj=o1, ea1=o3) == o3)

        # call from free variable
        self.assertTrue(concept_consumer1(o3, o4) == o4)

        # call from free variable
        self.assertTrue(concept_consumer1(o3, ea1=o4) == o4)

        # call from free variable
        self.assertTrue(concept_consumer1(obj=o3, ea1=o4) == o4)

    def test_concept_consumer2(self):
        o1 = Dummy(1)
        o2 = Dummy(2)
        o3 = Dummy(3)
        o4 = Dummy(4)
        CptA.link(o1, o2)

        # call from well defined concept
        self.assertTrue(concept_consumer2(o1) == o2)

        # call from well defined concept
        self.assertTrue(concept_consumer2(obj=o1) == o2)

        # call from well defined concept with argument replacement
        self.assertTrue(concept_consumer2(o1, o3) == o3)

        # call from well defined concept with named argument replacement
        self.assertTrue(concept_consumer2(o1, new_name=o3) == o3)

        # call from well defined concept with named argument replacement
        self.assertTrue(concept_consumer2(obj=o1, new_name=o3) == o3)

        # call from free variable
        self.assertTrue(concept_consumer2(o3, o4) == o4)

        # call from free variable
        self.assertTrue(concept_consumer2(o3, new_name=o4) == o4)

        # call from free variable
        self.assertTrue(concept_consumer2(obj=o3, new_name=o4) == o4)

    def test_concept_consumer3(self):
        o1 = Dummy(1)
        o2 = Dummy(2)
        o3 = Dummy(3)
        o4 = Dummy(4)
        o5 = Dummy(5)

        CptA.link(o1, o2)
        CptB.link(o3, o1, o4)

        # call from well defined concept
        self.assertTrue(concept_consumer3(o3) == o2)

        # call from well defined concept
        self.assertTrue(concept_consumer3(obj=o3) == o2)

        # call from well defined concept with argument replacement
        self.assertTrue(concept_consumer3(o3, o5) == o5)

        # call from well defined concept with named argument replacement
        self.assertTrue(concept_consumer3(o3, ea1=o5) == o5)

        # call from well defined concept with named argument replacement
        self.assertTrue(concept_consumer3(obj=o3, ea1=o5) == o5)

        # call from free variable
        self.assertTrue(concept_consumer3(o4, o5) == o5)

        # call from free variable
        self.assertTrue(concept_consumer3(o4, ea1=o5) == o5)

        # call from free variable
        self.assertTrue(concept_consumer3(obj=o4, ea1=o5) == o5)

    def test_concept_consumer4(self):
        o1 = Dummy(1)
        o2 = Dummy(2)
        o3 = Dummy(3)
        o4 = Dummy(4)
        o5 = Dummy(5)

        CptA.link(o1, o2)
        CptC.link(o2, o3)

        # call from well defined concept
        self.assertTrue(concept_consumer4(o1) == o3)

        # call from well defined concept
        self.assertTrue(concept_consumer4(obj=o1) == o3)

        # call from well defined concept with argument replacement
        self.assertTrue(concept_consumer4(o1, o4) == o4)

        # call from well defined concept with named argument replacement
        self.assertTrue(concept_consumer4(o1, ec1=o4) == o4)

        # call from well defined concept with named argument replacement
        self.assertTrue(concept_consumer4(obj=o1, ec1=o5) == o5)

        # call from free variable
        self.assertTrue(concept_consumer4(o4, o5) == o5)

        # call from free variable
        self.assertTrue(concept_consumer4(o4, ec1=o5) == o5)

        # call from free variable
        self.assertTrue(concept_consumer4(obj=o4, ec1=o5) == o5)


if __name__ == '__main__':
    unittest.main()
