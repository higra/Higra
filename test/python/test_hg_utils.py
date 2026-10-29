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
import numpy as np


class TestHGUtils(unittest.TestCase):

    def test_as_index_array(self):
        from higra.hg_utils import _as_index_array

        for values in ([], np.array([], dtype=float), [0, 2],
                       np.array([0, 2], dtype=">u8"), np.arange(6)[::2]):
            result = _as_index_array(values, "indices")
            np.testing.assert_array_equal(result, values)
            self.assertEqual(result.dtype, np.dtype(np.int64))
        for values in ([-1], [1.5], [True], [[0]], np.array([2**63], dtype=np.uint64)):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "indices"):
                _as_index_array(values, "indices")

    def test_validate_edge_weights(self):
        from higra.hg_utils import _validate_edge_weights

        for weights in ([1, 2], np.array([np.inf, -np.inf]),
                        np.ones((2, 3, 4), dtype=">f4"), np.empty((0, 2)),
                        np.arange(8, dtype=np.int16)[::2]):
            result = _validate_edge_weights(weights, len(weights))
            np.testing.assert_array_equal(result, weights)
            self.assertEqual(result.dtype, np.asarray(weights).dtype)
        for weights in (1, [1], [np.nan, 0], [True, False], np.ones(2, dtype=np.float16),
                        np.ones(2, dtype=complex), np.ones(2, dtype=object)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                _validate_edge_weights(weights, 2)

    def test_linearize_vertex_weights(self):
        g = hg.get_4_adjacency_graph((4, 5))

        r = hg.linearize_vertex_weights(np.ones((20,)), g, (4, 5))
        self.assertTrue(r.shape == (20,))

        r = hg.linearize_vertex_weights(np.ones((4, 5)), g, (4, 5))
        self.assertTrue(r.shape == (20,))

        r = hg.linearize_vertex_weights(np.ones((4, 5, 10, 12)), g, (4, 5))
        self.assertTrue(r.shape == (20, 10, 12))

        r = hg.linearize_vertex_weights(np.ones((20, 4, 5, 2, 3)), g, (4, 5))
        self.assertTrue(r.shape == (20, 4, 5, 2, 3))

        with self.assertRaises(ValueError):
            hg.linearize_vertex_weights(np.ones((5, 4)), g, (4, 5))

        with self.assertRaises(ValueError):
            hg.linearize_vertex_weights(np.ones((25,)), g, (4, 5))

        g = hg.get_4_adjacency_graph((4, 1))

        r = hg.linearize_vertex_weights(np.ones((4,)), g)
        self.assertTrue(r.shape == (4,))

        r = hg.linearize_vertex_weights(np.ones((4, 1)), g)
        self.assertTrue(r.shape == (4,))

        r = hg.linearize_vertex_weights(np.ones((4, 1, 10, 12)), g)
        self.assertTrue(r.shape == (4, 10, 12))

        r = hg.linearize_vertex_weights(np.ones((4, 4, 5, 2, 3)), g)
        self.assertTrue(r.shape == (4, 4, 5, 2, 3))

        with self.assertRaises(ValueError):
            hg.linearize_vertex_weights(np.ones((5, 4)), g, (4, 5))

        with self.assertRaises(ValueError):
            hg.linearize_vertex_weights(np.ones((25,)), g, (4, 5))

        g = hg.get_4_adjacency_graph((1, 4))

        r = hg.linearize_vertex_weights(np.ones((4,)), g)
        self.assertTrue(r.shape == (4,))

        r = hg.linearize_vertex_weights(np.ones((1, 4)), g)
        self.assertTrue(r.shape == (4,))

        r = hg.linearize_vertex_weights(np.ones((1, 4, 10, 12)), g)
        self.assertTrue(r.shape == (4, 10, 12))

        r = hg.linearize_vertex_weights(np.ones((4, 4, 5, 2, 3)), g)
        self.assertTrue(r.shape == (4, 4, 5, 2, 3))

        with self.assertRaises(ValueError):
            hg.linearize_vertex_weights(np.ones((5, 4)), g, (4, 5))

        with self.assertRaises(ValueError):
            hg.linearize_vertex_weights(np.ones((25,)), g, (4, 5))

    def test_delinearize_vertex_weights(self):
        g = hg.get_4_adjacency_graph((4, 5))

        r = hg.delinearize_vertex_weights(np.ones((20,)), g, (4, 5))
        self.assertTrue(r.shape == (4, 5))

        r = hg.delinearize_vertex_weights(np.ones((4, 5)), g, (4, 5))
        self.assertTrue(r.shape == (4, 5))

        r = hg.delinearize_vertex_weights(np.ones((4, 5, 10, 12)), g, (4, 5))
        self.assertTrue(r.shape == (4, 5, 10, 12))

        r = hg.delinearize_vertex_weights(np.ones((20, 4, 5, 2, 3)), g, (4, 5))
        self.assertTrue(r.shape == (4, 5, 4, 5, 2, 3))

        with self.assertRaises(ValueError):
            hg.delinearize_vertex_weights(np.ones((5, 4)), g, (4, 5))

        with self.assertRaises(ValueError):
            hg.delinearize_vertex_weights(np.ones((25,)), g, (4, 5))

        shape = (1, 4)
        g = hg.get_4_adjacency_graph(shape)
        r = hg.delinearize_vertex_weights(np.ones((4,)), g)
        self.assertTrue(r.shape == shape)

        r = hg.delinearize_vertex_weights(np.ones(shape), g)
        self.assertTrue(r.shape == shape)

        r = hg.delinearize_vertex_weights(np.ones((1, 4, 10, 12)), g)
        self.assertTrue(r.shape == (1, 4, 10, 12))

        r = hg.delinearize_vertex_weights(np.ones((4, 5, 2, 3)), g)
        self.assertTrue(r.shape == (1, 4, 5, 2, 3))

        with self.assertRaises(ValueError):
            hg.delinearize_vertex_weights(np.ones((5, 4)), g)

        with self.assertRaises(ValueError):
            hg.delinearize_vertex_weights(np.ones((8,)), g)

        shape = (4, 1)
        g = hg.get_4_adjacency_graph(shape)
        r = hg.delinearize_vertex_weights(np.ones((4,)), g)
        self.assertTrue(r.shape == shape)

        r = hg.delinearize_vertex_weights(np.ones(shape), g)
        self.assertTrue(r.shape == shape)

        r = hg.delinearize_vertex_weights(np.ones((4, 1, 10, 12)), g)
        self.assertTrue(r.shape == (4, 1, 10, 12))

        r = hg.delinearize_vertex_weights(np.ones((4, 5, 2, 3)), g)
        self.assertTrue(r.shape == (4, 1, 5, 2, 3))

        with self.assertRaises(ValueError):
            hg.delinearize_vertex_weights(np.ones((5, 4)), g)

        with self.assertRaises(ValueError):
            hg.delinearize_vertex_weights(np.ones((8,)), g)

        # test with numpy array shape:
        r = hg.delinearize_vertex_weights(np.ones((4, 5, 2)), shape=np.array((4, 5), dtype=np.int32))
        self.assertTrue(r.shape == (4, 5, 2))

    def test_shape_only_vertex_weights(self):
        shapes = ((2, 3), [2, 3], np.array((2, 3), dtype=np.int32),
                  (np.int64(2), np.uint64(3)))
        for dtype in (np.int16, np.uint64, np.float32, np.complex128, np.bool_):
            for shape in shapes:
                for trailing in ((), (4,), (2, 4)):
                    with self.subTest(dtype=dtype, shape=shape, trailing=trailing):
                        image = np.arange(6 * int(np.prod(trailing))).astype(dtype).reshape((2, 3) + trailing)
                        linear = hg.linearize_vertex_weights(image, shape=shape)
                        self.assertEqual(linear.shape, (6,) + trailing)
                        self.assertEqual(linear.dtype, image.dtype)
                        self.assertTrue(np.shares_memory(linear, image))
                        np.testing.assert_array_equal(linear, image.reshape((6,) + trailing))
                        restored = hg.delinearize_vertex_weights(linear, shape=shape)
                        self.assertEqual(restored.dtype, image.dtype)
                        np.testing.assert_array_equal(restored, image)
                        self.assertIs(hg.linearize_vertex_weights(linear, shape=shape), linear)
                        self.assertIs(hg.delinearize_vertex_weights(image, shape=shape), image)

    def test_vertex_weights_noncontiguous(self):
        base = np.arange(48, dtype=np.int16).reshape(4, 3, 4)
        for image in (base[::2, :, ::-1], base.transpose(1, 0, 2), np.asfortranarray(base)):
            shape = image.shape[:2]
            with self.subTest(strides=image.strides):
                linear = hg.linearize_vertex_weights(image, shape=shape)
                np.testing.assert_array_equal(linear, image.reshape((-1, 4)))
                self.assertEqual(linear.dtype, image.dtype)
                np.testing.assert_array_equal(hg.delinearize_vertex_weights(linear, shape=shape), image)
        linear = np.arange(48).reshape(6, 8)[:, ::2]
        self.assertIs(hg.linearize_vertex_weights(linear, shape=(2, 3)), linear)
        np.testing.assert_array_equal(hg.delinearize_vertex_weights(linear, shape=(2, 3)),
                                      linear.reshape(2, 3, 4))

    def test_vertex_weights_graph_shape_consistency(self):
        for graph in (hg.UndirectedGraph(6), hg.get_4_adjacency_graph((2, 3)),
                      hg.get_4_adjacency_implicit_graph((2, 3))):
            image = np.arange(24).reshape(2, 3, 4)
            linear = hg.linearize_vertex_weights(image, graph, (2, 3))
            np.testing.assert_array_equal(linear, image.reshape(6, 4))
            np.testing.assert_array_equal(hg.delinearize_vertex_weights(linear, graph, (2, 3)), image)
            alternate = image.reshape(3, 2, 4)
            np.testing.assert_array_equal(hg.linearize_vertex_weights(alternate, graph, (3, 2)), linear)
            np.testing.assert_array_equal(hg.delinearize_vertex_weights(linear, graph, (3, 2)), alternate)
            if hg.CptGridGraph.validate(graph):
                np.testing.assert_array_equal(hg.linearize_vertex_weights(image, graph), linear)
                np.testing.assert_array_equal(hg.delinearize_vertex_weights(linear, graph), image)
            for fun in (hg.linearize_vertex_weights, hg.delinearize_vertex_weights):
                for weights in (np.ones((2, 4, 3)), np.ones((8, 3))):
                    with self.subTest(fun=fun.__name__, graph=type(graph), weights=weights.shape):
                        with self.assertRaisesRegex(ValueError, "Graph vertex count 6"):
                            fun(weights, graph, (2, 4))
        graph = hg.UndirectedGraph(6)
        hg.CptGridGraph.link(graph, (2, 4))
        for fun in (hg.linearize_vertex_weights, hg.delinearize_vertex_weights):
            with self.assertRaisesRegex(ValueError, "Graph vertex count 6"):
                fun(np.ones((2, 4)), graph)

    def test_vertex_weights_prefix_precedence(self):
        for shape in ((4, 1), (1, 4), (1, 1), (4,)):
            image = np.arange(int(np.prod(shape)) * 6).reshape(shape + (2, 3))
            linear = hg.linearize_vertex_weights(image, shape=shape)
            np.testing.assert_array_equal(linear, image.reshape((int(np.prod(shape)), 2, 3)))
            self.assertIs(hg.delinearize_vertex_weights(image, shape=shape), image)
        linear = np.arange(144).reshape(6, 2, 3, 4)
        self.assertIs(hg.linearize_vertex_weights(linear, shape=(2, 3)), linear)
        np.testing.assert_array_equal(hg.delinearize_vertex_weights(linear, shape=(2, 3)),
                                      linear.reshape(2, 3, 2, 3, 4))

    def test_vertex_weights_passthrough_and_degenerate_shapes(self):
        for weights in (np.array(5), np.ones((2, 3)), object()):
            for fun in (hg.linearize_vertex_weights, hg.delinearize_vertex_weights):
                self.assertIs(fun(weights), weights)
                self.assertIs(fun(weights, hg.UndirectedGraph(7)), weights)
        for graph in (None, hg.UndirectedGraph(1)):
            scalar = np.array(5, dtype=np.int16)
            np.testing.assert_array_equal(hg.linearize_vertex_weights(scalar, graph, ()), [5])
            self.assertIs(hg.delinearize_vertex_weights(scalar, graph, ()), scalar)
        for shape in ((0, 3), (2, 0)):
            for graph in (None, hg.UndirectedGraph(0)):
                image = np.empty(shape + (4,), dtype=np.int16)
                linear = hg.linearize_vertex_weights(image, graph, shape)
                self.assertEqual(linear.shape, (0, 4))
                restored = hg.delinearize_vertex_weights(linear, graph, shape)
                self.assertEqual(restored.shape, image.shape)
                self.assertEqual(restored.dtype, image.dtype)

    def test_vertex_weights_invalid_inputs(self):
        for fun in (hg.linearize_vertex_weights, hg.delinearize_vertex_weights):
            for weights in (np.array(5), np.ones(7), np.ones((3, 2)), np.ones((2,))):
                with self.subTest(fun=fun.__name__, weights=weights.shape):
                    with self.assertRaisesRegex(ValueError, "Vertex weights shape"):
                        fun(weights, shape=(2, 3))
            for shape in ((-2, -3), (2, -1)):
                with self.assertRaisesRegex(ValueError, "non-negative"):
                    fun(np.ones(6), shape=shape)
            for shape in (3, (2.0, 3), ("2", 3), ((2, 3),), np.array([[2, 3]])):
                with self.subTest(fun=fun.__name__, shape=shape):
                    with self.assertRaises(TypeError):
                        fun(np.ones(6), shape=shape)
            # The product is 2**64, rather than zero through fixed-width overflow.
            with self.assertRaisesRegex(ValueError, "Vertex weights shape"):
                fun(np.empty(0), shape=(np.int64(2**32), np.int64(2**32)))
        with self.assertRaisesRegex(ValueError, "6 vertices"):
            hg.delinearize_vertex_weights(np.ones(7), shape=(2, 3))

    def test_common_type(self):
        a_bool = np.zeros((1, 1), dtype=np.bool_)
        a_uint16 = np.zeros((1, 1), dtype=np.uint16)
        a_int8 = np.zeros((1, 1), dtype=np.int8)
        a_uint64 = np.zeros((1, 1), dtype=np.uint64)

        self.assertTrue(hg.common_type(a_uint16, a_int8, a_uint64) == np.int64)
        self.assertTrue(hg.common_type(a_bool, a_uint16, a_int8) == np.int32)
        self.assertTrue(hg.common_type(a_uint16, a_uint16) == np.uint16)

        self.assertTrue(hg.common_type(a_uint16, a_int8, a_uint64, safety_level='overflow') == np.float64)
        self.assertTrue(hg.common_type(a_bool, a_uint16, a_int8, safety_level='overflow') == np.float64)
        self.assertTrue(hg.common_type(a_uint16, a_uint16, safety_level='overflow') == np.float64)

    def test_cast_to_common_type(self):
        a_uint16 = np.zeros((1, 1), dtype=np.uint16)
        a_int8 = np.zeros((1, 1), dtype=np.int8)
        a_uint64 = np.zeros((1, 1), dtype=np.uint64)
        a_int64 = np.zeros((1, 1), dtype=np.int64)

        a, b, c, d = hg.cast_to_common_type(a_uint16, a_int8, a_int64, a_uint64)
        self.assertTrue(a.dtype == np.int64)
        self.assertTrue(b.dtype == np.int64)
        self.assertTrue(c.dtype == np.int64)
        self.assertTrue(d.dtype == np.int64)

        self.assertTrue(id(c) == id(a_int64))

    def test_type_consistency(self):
        tree = hg.Tree((2, 2, 2))
        for t in (np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64):
            a = np.zeros(3, dtype=t)
            res = hg.reconstruct_leaf_data(tree, a)
            self.assertTrue(a.dtype == res.dtype)

    def test_has_method(self):

        class Op:
            pass

        a = Op()
        self.assertFalse(hg.has_method(a, "foo"))

        a.foo = 1
        self.assertFalse(hg.has_method(a, "foo"))

        a.foo = lambda x: x
        self.assertTrue(hg.has_method(a, "foo"))
