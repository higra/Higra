############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import inspect
from pathlib import Path
import subprocess
import sys
import unittest
import warnings

import numpy as np
import higra as hg


class MyException(Exception):
    pass


def accept_everything(dummy, attr1=1):
    return attr1


@hg.argument_helper(hg.CptGridGraph)
def accept_RegularGraph2d(graph, shape):
    if type(graph) != hg.RegularGraph2d:
        raise MyException("kaput")
    return 2


class TestDataCache(unittest.TestCase):
    def decorate(self, fun):
        with self.assertWarnsRegex(FutureWarning, "auto_cache.*no-op.*next release"):
            return hg.auto_cache(fun)

    def test_recompute_mutable_input_and_output(self):
        calls = []

        def total(array, scale=1):
            """A useful docstring."""
            calls.append(scale)
            return np.array([array.sum() * scale])

        wrapped = self.decorate(total)
        self.assertEqual(inspect.signature(wrapped), inspect.signature(total))
        self.assertEqual(wrapped.__doc__, total.__doc__)
        self.assertEqual(wrapped.__name__, total.__name__)
        self.assertIs(wrapped.original, total)
        array = np.zeros(3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wrapped(array)[0] = 99
            np.testing.assert_array_equal(wrapped(array=array, scale=1), [0])
            array[0] = 7
            np.testing.assert_array_equal(wrapped(array), [7])
        self.assertEqual(caught, [])
        self.assertEqual(calls, [1, 1, 1])

    def test_exceptions_execute_once(self):
        for error in (TypeError("application failure"), ValueError("failure"), MyException()):
            with self.subTest(error=type(error)):
                calls = []

                def failing(obj):
                    calls.append(obj)
                    raise error

                wrapped = self.decorate(failing)
                with self.assertRaises(type(error)) as caught:
                    wrapped(np.zeros(1))
                self.assertIs(caught.exception, error)
                self.assertEqual(len(calls), 1)

    def test_unrestricted_signatures_and_nonweak_inputs(self):
        def flexible(first=2, /, *values, scale=3, **options):
            return (first + sum(values)) * scale + options.get("offset", 0)

        wrapped = self.decorate(flexible)
        self.assertEqual(wrapped(), 6)
        self.assertEqual(wrapped(1, 2, scale=4, offset=5), 17)
        self.assertEqual(self.decorate(lambda x: sum(x))([1, 2]), 3)
        self.assertEqual(self.decorate(lambda: 8)(), 8)

    def test_legacy_keywords(self):
        wrapped = self.decorate(lambda value=3: value)
        for name, value in self.controls().items():
            with self.subTest(name=name):
                with self.assertWarnsRegex(FutureWarning, name + ".*ignored.*next release"):
                    self.assertEqual(wrapped(**{name: value}), 3)

    @staticmethod
    def controls():
        return dict(no_cache=False, force_recompute=False, attribute_name=None, data_cache=None)

    def test_declared_control_names_keep_meaning(self):
        def declared(no_cache, force_recompute, attribute_name, data_cache):
            return no_cache, force_recompute, attribute_name, data_cache

        wrapped = self.decorate(declared)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(wrapped(1, 2, attribute_name=3, data_cache=4), (1, 2, 3, 4))
        self.assertEqual(caught, [])

    def test_all_formerly_cached_functions_accept_controls(self):
        from higra.attribute import tree_attributes, graph_attributes

        functions = [fun for module in (tree_attributes, graph_attributes)
                     for fun in vars(module).values()
                     if callable(fun) and getattr(fun, "_legacy_cache_keywords", False)]
        self.assertEqual(len(functions), 25)
        for fun in functions:
            for name, value in self.controls().items():
                with self.subTest(function=fun.__name__, control=name):
                    with self.assertWarnsRegex(FutureWarning, name) as warning:
                        with self.assertRaisesRegex(TypeError, "missing .*required positional argument"):
                            fun(**{name: value})
                    self.assertEqual(warning.filename, __file__)

    def test_explicit_concept_cache(self):
        graph = hg.get_4_adjacency_graph((1, 2))
        tree = hg.Tree((2, 2, 2))
        hg.CptHierarchy.link(tree, graph)
        cache = hg.DataCache()
        # The custom store overrides the stored leaf graph during construction.
        other_graph = hg.get_4_adjacency_graph((2, 1))
        cache.get_data(tree)["leaf_graph"] = other_graph
        self.assertIs(hg.CptHierarchy.construct(tree, data_cache=cache)["leaf_graph"], other_graph)

    def test_state_is_always_disabled(self):
        for state in (False, True):
            with self.assertWarnsRegex(FutureWarning, "set_auto_cache_state.*no-op.*next release"):
                self.assertIsNone(hg.set_auto_cache_state(state))
            with self.assertWarnsRegex(FutureWarning, "get_auto_cache_state.*next release"):
                self.assertIs(hg.get_auto_cache_state(), False)
        with self.assertWarns(FutureWarning):
            with self.assertRaises(TypeError):
                hg.set_auto_cache_state(1)

    def test_clear_preserves_metadata_and_explicit_data(self):
        graph = hg.get_4_adjacency_graph((1, 2))
        tree = hg.Tree((2, 2, 2))
        hg.CptHierarchy.link(tree, graph)
        hg.set_attribute(tree, "user_value", 42)
        hg.add_tag(tree, "user_tag")
        cache = hg.DataCache()
        stored = cache.get_data(tree)
        stored.update(user_value=7, data_auto_cache={"old": 1})
        preprocessed = tree.lowest_common_ancestor_preprocess()
        for options in ({}, {"function": "anything"}, {"reference_object": tree},
                        {"function": object(), "reference_object": tree, "data_cache": cache}):
            with self.assertWarnsRegex(FutureWarning, "clear_auto_cache.*no-op.*next release"):
                self.assertIsNone(hg.clear_auto_cache(**options))
        self.assertEqual(stored, dict(user_value=7, data_auto_cache={"old": 1}))
        self.assertEqual(hg.get_attribute(tree, "user_value"), 42)
        self.assertTrue(hg.has_tag(tree, "user_tag"))
        self.assertIs(hg.CptHierarchy.get_leaf_graph(tree), graph)
        self.assertIs(tree.lowest_common_ancestor_preprocess(), preprocessed)
        self.assertIsNot(tree.lowest_common_ancestor_preprocess(force_recompute=True), preprocessed)

    def test_argument_helper_composition_both_orders(self):
        graph = hg.get_4_adjacency_graph((2, 3))
        for cache_outside in (False, True):
            calls = []

            def shape_of(graph, shape=None):
                calls.append(graph)
                return shape

            helper = hg.argument_helper(hg.CptGridGraph)
            wrapped = (self.decorate(helper(shape_of)) if cache_outside
                       else helper(self.decorate(shape_of)))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertEqual(tuple(wrapped(graph)), (2, 3))
                self.assertEqual(tuple(wrapped(graph=graph)), (2, 3))
            self.assertEqual(caught, [])
            for name, value in self.controls().items():
                with self.assertWarnsRegex(FutureWarning, name):
                    self.assertEqual(tuple(wrapped(graph, **{name: value})), (2, 3))
            self.assertEqual(len(calls), 6)
            self.assertEqual(inspect.signature(wrapped), inspect.signature(shape_of))

    def test_declared_data_cache_with_argument_helper(self):
        @hg.argument_helper(hg.CptGridGraph)
        def shape_of(graph, data_cache, shape=None):
            return data_cache, shape

        wrapped = self.decorate(shape_of)
        graph = hg.get_4_adjacency_graph((2, 3))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(wrapped(graph, 42), (42, (2, 3)))
        self.assertEqual(caught, [])

    def test_tree_attribute_mutation_and_legacy_controls(self):
        tree = hg.Tree(np.array([2, 2, 2], dtype=np.int64))
        for dtype in (np.int32, np.float32, np.float64):
            areas = np.array([1, 1], dtype=dtype)
            result = hg.attribute_area(tree, vertex_area=areas)
            np.testing.assert_array_equal(result, [1, 1, 2])
            result[:] = 99
            areas[0] = 4
            result = hg.attribute_area(tree, vertex_area=areas)
            np.testing.assert_array_equal(result, [4, 1, 5])
            self.assertEqual(result.dtype, dtype)
            for name, value in self.controls().items():
                with self.assertWarnsRegex(FutureWarning, name):
                    np.testing.assert_array_equal(
                        hg.attribute_area(tree, vertex_area=areas, **{name: value}), [4, 1, 5])
        with self.assertWarns(FutureWarning):
            np.testing.assert_array_equal(hg.attribute_depth(tree, no_cache=False), [1, 1, 0])

    def test_graph_attribute_mutation_and_shapes(self):
        for graph in (hg.get_4_adjacency_graph((1, 2)),
                      hg.get_4_adjacency_implicit_graph((1, 2))):
            # Edge accumulation is bound for explicit graphs; implicit grids use border metadata.
            if isinstance(graph, hg.UndirectedGraph):
                hg.set_attribute(graph, "no_border_vertex_out_degree", None)
                lengths = np.array([1.])
                result = hg.attribute_vertex_perimeter(graph, lengths)
                self.assertEqual(result.shape, (1, 2))
                np.testing.assert_array_equal(result, [[1, 1]])
                result[:] = 99
                lengths[0] = 4
                np.testing.assert_array_equal(hg.attribute_vertex_perimeter(graph, lengths), [[4, 4]])
            hg.set_attribute(graph, "no_border_vertex_out_degree", 8)
            np.testing.assert_array_equal(hg.attribute_vertex_perimeter(graph, edge_length=np.ones(1)), [[8, 8]])
            for name, value in self.controls().items():
                with self.assertWarnsRegex(FutureWarning, name):
                    coordinates = hg.attribute_vertex_coordinates(graph, **{name: value})
                np.testing.assert_array_equal(coordinates, [[[0, 0], [0, 1]]])
            vertex_area = hg.attribute_vertex_area(graph)
            vertex_area[:] = 99
            np.testing.assert_array_equal(hg.attribute_vertex_area(graph), [[1, 1]])

    def test_import_and_ordinary_calls_are_quiet(self):
        subprocess.run([sys.executable, "-Werror::FutureWarning", "-c",
                        "import higra as hg; g = hg.get_4_adjacency_graph((2, 3)); "
                        "t, a = hg.component_tree_max_tree(g, __import__('numpy').arange(6)); "
                        "hg.attribute_area(t); hg.attribute_vertex_coordinates(g)"],
                       cwd=Path(hg.__file__).resolve().parent.parent, check=True)

    def test_argument_helper_accept_everything(self):
        for value in (2, (3, 2), {3, 2}, {"toto": 42}):
            self.assertEqual(accept_everything(value), 1)

    def test_argument_helper_accept_RegularGraph2d(self):
        graph = hg.get_4_adjacency_implicit_graph((2, 3))
        self.assertEqual(accept_RegularGraph2d(graph), 2)
        self.assertRaises(MyException, accept_RegularGraph2d, (4, 5), (2, 3))


if __name__ == '__main__':
    unittest.main()
