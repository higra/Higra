############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import gc
import weakref
import unittest
import higra as hg
import numpy as np


class TestUndirectedGraph(unittest.TestCase):

    @staticmethod
    def test_graph():
        g = hg.UndirectedGraph(4)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(0, 2)
        return g

    def test_add_vertex(self):
        g = hg.UndirectedGraph()
        self.assertTrue(g.num_vertices() == 0)
        self.assertTrue(g.add_vertex() == 0)
        self.assertTrue(g.num_vertices() == 1)
        self.assertTrue(g.add_vertex() == 1)
        self.assertTrue(g.num_vertices() == 2)

        g = hg.UndirectedGraph(3)
        self.assertTrue(g.num_vertices() == 3)

    def test_add_vertices(self):
        g = hg.UndirectedGraph()
        self.assertTrue(g.num_vertices() == 0)
        g.add_vertices(3)
        self.assertTrue(g.num_vertices() == 3)
        g.add_vertices(2)
        self.assertTrue(g.num_vertices() == 5)

    def test_add_edge(self):
        g = hg.UndirectedGraph(3)
        self.assertTrue(g.num_edges() == 0)
        g.add_edge(0, 1)
        self.assertTrue(g.num_edges() == 1)

        # parallel edge allowed
        g.add_edge(0, 1)
        self.assertTrue(g.num_edges() == 2)

        # still parallel edge allowed
        g.add_edge(1, 0)
        self.assertTrue(g.num_edges() == 3)

        g.add_edge(0, 2)
        self.assertTrue(g.num_edges() == 4)

    def test_add_edges(self):
        g = hg.UndirectedGraph(3)
        g.add_edge(0, 1)
        g.add_edge(0, 2)

        g2 = hg.UndirectedGraph(3)
        g2.add_edges((0, 0), (1, 2))

        self.assertTrue(g2.num_edges() == 2)

        for i in range(g2.num_edges()):
            self.assertTrue(g.edge_from_index(i) == g2.edge_from_index(i))

    def test_dynamic_attributes(self):
        g = TestUndirectedGraph.test_graph()
        g.new_attribute = 42
        self.assertTrue(g.new_attribute == 42)

    def test_vertex_iterator(self):
        g = TestUndirectedGraph.test_graph()
        vref = [0, 1, 2, 3];
        vtest = [];

        for v in g.vertices():
            vtest.append(v)

        self.assertTrue(vtest == vref)

    def test_edge_iterator(self):
        g = TestUndirectedGraph.test_graph()
        ref = [(0, 1), (1, 2), (0, 2)]
        test = []

        for e in g.edges():
            test.append((g.source(e), g.target(e)))

        self.assertTrue(test == ref)

    def test_edge_list(self):
        g = TestUndirectedGraph.test_graph()
        ref_sources = (0, 1, 0)
        ref_targets = (1, 2, 2)

        sources = g.sources()
        self.assertTrue(np.all(ref_sources == sources))

        targets = g.targets()
        self.assertTrue(np.all(ref_targets == targets))

        sources, targets = g.edge_list()
        self.assertTrue(np.all(ref_sources == sources))
        self.assertTrue(np.all(ref_targets == targets))

    def test_out_edge_iterator(self):
        g = TestUndirectedGraph.test_graph()
        ref = [[(0, 1), (0, 2)],
               [(1, 0), (1, 2)],
               [(2, 1), (2, 0)],
               []]
        test = []
        for v in g.vertices():
            test.append([])
            for e in g.out_edges(v):
                test[v].append((e[0], e[1]))

        self.assertTrue(test == ref)

    def test_in_edge_iterator(self):
        g = TestUndirectedGraph.test_graph()
        ref = [[(1, 0), (2, 0)],
               [(0, 1), (2, 1)],
               [(1, 2), (0, 2)],
               []]
        test = []
        for v in g.vertices():
            test.append([])
            for e in g.in_edges(v):
                test[v].append((e[0], e[1]))

        self.assertTrue(test == ref)

    def test_adjacent_vertex_iterator(self):
        g = TestUndirectedGraph.test_graph()
        ref = [[1, 2],
               [0, 2],
               [1, 0],
               []]
        test = []
        for v in g.vertices():
            test.append([])
            for av in g.adjacent_vertices(v):
                test[v].append(av)

        self.assertTrue(test == ref)

    def test_degrees(self):
        g = TestUndirectedGraph.test_graph()
        self.assertTrue(g.degree(0) == 2)
        self.assertTrue(g.out_degree(0) == 2)
        self.assertTrue(g.in_degree(0) == 2)

        self.assertTrue(g.degree(1) == 2)
        self.assertTrue(g.out_degree(1) == 2)
        self.assertTrue(g.in_degree(1) == 2)

        self.assertTrue(g.degree(2) == 2)
        self.assertTrue(g.out_degree(2) == 2)
        self.assertTrue(g.in_degree(2) == 2)

        self.assertTrue(g.degree(3) == 0)
        self.assertTrue(g.out_degree(3) == 0)
        self.assertTrue(g.in_degree(3) == 0)

        indices = np.asarray(((0, 3), (1, 2)))
        ref = np.asarray(((2, 0), (2, 2)))
        self.assertTrue(np.allclose(g.degree(indices), ref))
        self.assertTrue(np.allclose(g.in_degree(indices), ref))
        self.assertTrue(np.allclose(g.out_degree(indices), ref))

    def test_edge_index_iterator(self):
        g = TestUndirectedGraph.test_graph()
        ref = [0, 1, 2]
        test = []

        for e in g.edges():
            test.append(g.index(e))

        self.assertTrue(test == ref)

    def test_out_edge_iterator(self):
        g = TestUndirectedGraph.test_graph()
        ref = [[0, 2],
               [0, 1],
               [1, 2],
               []]
        test = []
        for v in g.vertices():
            test.append([])
            for e in g.out_edges(v):
                test[v].append(e[2])

        self.assertTrue(test == ref)

    def test_in_edge_index_iterator(self):
        g = TestUndirectedGraph.test_graph()
        ref = [[0, 2],
               [0, 1],
               [1, 2],
               []]
        test = []
        for v in g.vertices():
            test.append([])
            for e in g.in_edges(v):
                test[v].append(e[2])

        self.assertTrue(test == ref)

    def assert_topology_consistent(self, graph):
        sources, targets = graph._sources(), graph._targets()
        self.assertEqual(list(graph.edges()),
                         [(int(s), int(t), i) for i, (s, t) in enumerate(zip(sources, targets))])
        for vertex in graph.vertices():
            incident = [i for i, (s, t) in enumerate(zip(sources, targets))
                        if s == vertex or t == vertex]
            adjacent = [int(t if s == vertex else s)
                        for s, t in zip(sources, targets) if s == vertex or t == vertex]
            self.assertEqual(sorted(e[2] for e in graph.out_edges(vertex)), incident)
            self.assertEqual(sorted(e[2] for e in graph.in_edges(vertex)), incident)
            self.assertEqual(sorted(graph.adjacent_vertices(vertex)), sorted(adjacent))
            self.assertEqual(graph.degree(vertex), len(incident))

    def assert_readonly(self, array):
        self.assertIs(type(array), np.ndarray)
        self.assertFalse(array.flags.owndata)
        self.assertFalse(array.flags.writeable)
        self.assertIsNotNone(array.base)
        with self.assertRaises(ValueError):
            array.setflags(write=True)
        with self.assertRaises(ValueError):
            array.flags.writeable = True
        with self.assertRaises(ValueError):
            array.flags['WRITEABLE'] = True
        if array.size:
            with self.assertRaises(ValueError):
                array[0] = 2
        self.assertTrue(memoryview(array).readonly)
        # Exercise a consumer explicitly asking NumPy for a writable buffer.
        import ctypes
        get_buffer = ctypes.pythonapi.PyObject_GetBuffer
        get_buffer.argtypes = (ctypes.py_object, ctypes.c_void_p, ctypes.c_int)
        get_buffer.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(128)  # larger than Py_buffer
        with self.assertRaises((ValueError, BufferError)):
            get_buffer(array, buffer, 0x0019)  # PyBUF_STRIDES | PyBUF_WRITABLE

    def test_endpoint_array_contract(self):
        for graph_type in (hg.UndirectedGraph, hg.UndirectedGraphOptimizedDelete):
            for empty in (True, False):
                with self.subTest(graph_type=graph_type, empty=empty):
                    g = graph_type(4, 16, 2)
                    if not empty:
                        g.add_edges((0, 1, 0, 2), (1, 2, 1, 2))
                    accessors = [g._sources, g._targets]
                    if graph_type is hg.UndirectedGraph:
                        accessors += [g.sources, g.targets]
                        self.assertIs(type(g.edge_list()), tuple)
                        for a in g.edge_list():
                            self.assert_readonly(a)
                    for accessor in accessors:
                        a = accessor()
                        self.assertEqual(a.dtype, np.dtype(np.int64))
                        self.assertEqual(a.shape, (g.num_edges(),))
                        self.assertEqual(a.strides, (3 * a.itemsize,))
                        self.assert_readonly(a)
                        for view in (a[:], a[::-1], a[::2], np.asarray(memoryview(a))):
                            self.assert_readonly(view)
                        if not empty:
                            self.assertTrue(np.shares_memory(a, accessor()))
                    self.assert_topology_consistent(g)

    def test_endpoint_generations_and_inplace_edits(self):
        for graph_type in (hg.UndirectedGraph, hg.UndirectedGraphOptimizedDelete):
            for reservation in (0, 16):
                with self.subTest(graph_type=graph_type, reservation=reservation):
                    g = graph_type(4, reservation)
                    g.add_edge(0, 1)
                    early = g._sources(), g._targets()
                    g.set_edge(0, 2, 3)
                    self.assertEqual((early[0][0], early[1][0]), (2, 3))
                    g.add_vertex()
                    self.assertTrue(np.shares_memory(early[0], g._sources()))
                    generations = [(early, tuple(a.copy() for a in early))]
                    for count in (32, 128, 512, 2048):
                        while g.num_edges() < count:
                            g.add_edge(0, g.num_edges() % 5)
                        current = g._sources(), g._targets()
                        for a in current:
                            self.assert_readonly(a)
                        self.assertTrue(np.shares_memory(current[0], g._sources()))
                        self.assertTrue(np.shares_memory(current[1], g._targets()))
                        generations.append((current, tuple(a.copy() for a in current)))
                    self.assertFalse(np.shares_memory(early[0], g._sources()))
                    g.set_edge(0, 0, 1)
                    self.assertEqual((current[0][0], current[1][0]), (0, 1))
                    g.remove_edge(0)
                    self.assertEqual((current[0][0], current[1][0]), (-1, -1))
                    for arrays, expected in generations[:-1]:
                        for a, e in zip(arrays, expected):
                            np.testing.assert_array_equal(a, e)
                            self.assertEqual(a.shape, e.shape)
                    self.assert_topology_consistent(g)

    def test_endpoint_owners_and_derived_views(self):
        for graph_type in (hg.UndirectedGraph, hg.UndirectedGraphOptimizedDelete):
            for empty in (True, False):
                with self.subTest(graph_type=graph_type, empty=empty):
                    g = graph_type(4, 4)
                    if not empty:
                        g.add_edges((0, 1, 2, 0), (1, 2, 3, 3))
                    try:
                        graph_ref = weakref.ref(g)
                    except TypeError:
                        graph_ref = None  # this variant need not support Python weakrefs
                    sources, targets = g._sources(), g._targets()
                    expected = sources.copy(), targets.copy()
                    views = (sources[::2], sources[::-1], targets[1:], targets[::-2])
                    expected_views = tuple(a.copy() for a in views)
                    buffers = memoryview(sources), memoryview(targets)
                    for _ in range(256):
                        g.add_edge(0, 3)
                    del sources, targets, g
                    gc.collect()
                    if graph_ref is not None:
                        self.assertIsNotNone(graph_ref())
                    for i in range(2):
                        np.testing.assert_array_equal(buffers[i], expected[i])
                    for i in range(len(views)):
                        np.testing.assert_array_equal(views[i], expected_views[i])
                        self.assert_readonly(views[i])
                    del views, buffers
                    gc.collect()
                    if graph_ref is not None:
                        self.assertIsNone(graph_ref())

    def test_endpoint_self_append_inputs_and_consumers(self):
        for graph_type in (hg.UndirectedGraph, hg.UndirectedGraphOptimizedDelete):
            for dtype in (np.int32, np.uint32, np.int64, np.uint64):
                with self.subTest(graph_type=graph_type, dtype=dtype):
                    g = graph_type(4)
                    g.add_edges(np.array([0, 1, 2, 0], dtype=dtype),
                                np.array([1, 2, 2, 1], dtype=dtype))
                    if graph_type is hg.UndirectedGraph:
                        s, t = g.edge_list()
                        g.add_edges(*g.edge_list())
                    else:
                        s, t = g._sources(), g._targets()
                        g.add_edges(s, t)
                    np.testing.assert_array_equal(g._sources(), np.tile(s, 2))
                    np.testing.assert_array_equal(g._targets(), np.tile(t, 2))
                    self.assert_topology_consistent(g)
                    before = tuple(g.edges())
                    for action in (lambda: g.add_edge(-1, 1),
                                   lambda: g.add_edges(np.array([0, 4]), np.array([1, 2])),
                                   lambda: g.add_edges(np.array([0, 1]), np.array([1])),
                                   lambda: g.set_edge(0, 0, 4),
                                   lambda: g.set_edge(100, 0, 1),
                                   lambda: g.remove_edge(100)):
                        with self.assertRaises(RuntimeError):
                            action()
                        self.assertEqual(tuple(g.edges()), before)
                        self.assert_topology_consistent(g)
                    np.testing.assert_array_equal(s[[3, 1]], [0, 1])
        g = self.test_graph()
        sparse = hg.undirected_graph_2_adjacency_matrix(g)
        dense = hg.undirected_graph_2_adjacency_matrix(g, sparse=False)
        np.testing.assert_array_equal(sparse.toarray(), dense)
        subgraph = hg.subgraph(g, np.array([2, 0]))
        np.testing.assert_array_equal(subgraph.sources(), g.sources()[[2, 0]])
        np.testing.assert_array_equal(subgraph.targets(), g.targets()[[2, 0]])

    def test_pickle(self):
        import pickle
        g = TestUndirectedGraph.test_graph()
        hg.set_attribute(g, "test", (1, 2, 3))
        hg.add_tag(g, "foo")

        data = pickle.dumps(g)
        g2 = pickle.loads(data)

        self.assertTrue(g.num_vertices() == g2.num_vertices())
        gs, gt = g.edge_list()
        g2s, g2t = g2.edge_list()
        self.assertTrue(np.all(gs == g2s))
        self.assertTrue(np.all(gt == g2t))

        self.assertTrue(hg.get_attribute(g, "test") == hg.get_attribute(g2, "test"))
        self.assertTrue(g.test == g2.test)
        self.assertTrue(hg.has_tag(g2, "foo"))


if __name__ == '__main__':
    unittest.main()
