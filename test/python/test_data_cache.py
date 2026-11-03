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
        with self.assertWarnsRegex(FutureWarning, "DataCache.*removed.*next release"):
            cache = hg.DataCache()
        # The custom store overrides the stored leaf graph during construction.
        other_graph = hg.get_4_adjacency_graph((2, 1))
        cache.get_data(tree)["leaf_graph"] = other_graph
        with self.assertWarnsRegex(FutureWarning, "construct data_cache.*removed.*next release"):
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
        tree.user_value = 42
        tree.__dict__.setdefault("__higra_tags__", set()).update({"user_tag"})
        with self.assertWarnsRegex(FutureWarning, "DataCache.*removed.*next release"):
            cache = hg.DataCache()
        stored = cache.get_data(tree)
        stored.update(user_value=7, data_auto_cache={"old": 1})
        preprocessed = tree.lowest_common_ancestor_preprocess()
        for options in ({}, {"function": "anything"}, {"reference_object": tree},
                        {"function": object(), "reference_object": tree, "data_cache": cache}):
            with self.assertWarnsRegex(FutureWarning, "clear_auto_cache.*no-op.*next release"):
                self.assertIsNone(hg.clear_auto_cache(**options))
        self.assertEqual(stored, dict(user_value=7, data_auto_cache={"old": 1}))
        self.assertEqual(getattr(tree, "user_value", None), 42)
        self.assertTrue("user_tag" in tree.__higra_tags__)
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
                graph.no_border_vertex_out_degree = None
                lengths = np.array([1.])
                result = hg.attribute_vertex_perimeter(graph, lengths)
                self.assertEqual(result.shape, (1, 2))
                np.testing.assert_array_equal(result, [[1, 1]])
                result[:] = 99
                lengths[0] = 4
                np.testing.assert_array_equal(hg.attribute_vertex_perimeter(graph, lengths), [[4, 4]])
            graph.no_border_vertex_out_degree = 8
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


class TestDynamicMetadata(unittest.TestCase):
    class Owner:
        # Dynamic storage does not require weak-reference support.
        __slots__ = ("__dict__",)

    def deprecated(self, name, *args, **kwargs):
        with self.assertWarnsRegex(FutureWarning, name + ".*private.*next release") as caught:
            result = getattr(hg, name)(*args, **kwargs)
        self.assertEqual(caught.filename, __file__)
        return result

    def test_fields_and_public_helpers(self):
        obj = self.Owner()
        self.assertIsNone(self.deprecated("get_attribute", obj, "value"))
        self.assertFalse(self.deprecated("has_tag", obj, "tag"))
        self.assertEqual(vars(obj), {})
        array = np.arange(3)
        self.deprecated("set_attribute", obj, "value", array, insert_dynamic=False)
        self.assertIs(obj.value, array)
        obj.value = 42
        self.assertEqual(self.deprecated("get_attribute", obj, "value"), 42)
        self.deprecated("add_tag", obj, "tag")
        tags = self.deprecated("get_tags", obj)
        self.assertIs(tags, obj.__higra_tags__)
        self.assertEqual(self.deprecated("list_attributes", obj), list(vars(obj)))
        self.deprecated("remove_tag", obj, "tag")
        with self.assertWarns(FutureWarning), self.assertRaises(KeyError):
            hg.remove_tag(obj, "tag")
        del obj.value
        self.assertIsNone(self.deprecated("get_attribute", obj, "value"))
        self.assertNotIn("__higra_attributes__", vars(obj))

    def test_unsupported_owners(self):
        for obj in (np.arange(2), np.int64(1), 1, [], object()):
            with self.subTest(owner=type(obj)):
                self.assertIsNone(self.deprecated("get_attribute", obj, "x"))
                self.assertEqual(self.deprecated("list_attributes", obj), [])
                self.assertFalse(self.deprecated("has_tag", obj, hg.CptGridGraph))
                self.assertFalse(hg.CptGridGraph.validate(obj))
                for name, args in (("set_attribute", ("x", 1)), ("get_tags", ()),
                                   ("add_tag", ("x",)), ("remove_tag", ("x",))):
                    with self.assertWarns(FutureWarning), self.assertRaisesRegex(TypeError, "dynamic attributes"):
                        getattr(hg, name)(obj, *args)
                with self.assertRaisesRegex(TypeError, "dynamic attributes"):
                    hg.CptGridGraph.link(obj, (2,))

    def test_clearing_preserves_all_state(self):
        graph = hg.get_4_adjacency_graph((1, 2))
        tree = hg.Tree((2, 2, 2))
        hg.CptHierarchy.link(tree, graph)
        tree.array = np.arange(3)
        lca = tree.lowest_common_ancestor_preprocess()
        with self.assertWarns(FutureWarning):
            store = hg.DataCache()
        stored = store.get_data(tree)
        stored["value"] = tree.array
        before = dict(vars(tree))
        for args in ((tree,), (tree, "array"), (tree, "array", "array", "missing"), (np.zeros(1),)):
            with self.assertWarnsRegex(FutureWarning, "no-op.*removed.*next release"):
                self.assertIsNone(hg.clear_attributes(*args))
        with self.assertWarnsRegex(FutureWarning, "no-op.*removed.*next release"):
            self.assertIsNone(hg.clear_all_attributes())
        self.assertEqual(set(before), set(vars(tree)))
        for key, value in before.items():
            self.assertIs(vars(tree)[key], value)
        self.assertIs(stored["value"], tree.array)
        self.assertIs(tree.lowest_common_ancestor_preprocess(), lca)
        self.assertIs(hg.CptHierarchy.get_leaf_graph(tree), graph)

    def test_standalone_storage_deprecated(self):
        for cls in (hg.DataCache, hg.WeakKeyDictionary):
            with self.assertWarnsRegex(FutureWarning, cls.__name__ + ".*removed.*next release") as caught:
                store = cls()
            self.assertEqual(caught.filename, __file__)
            key = np.arange(3)
            if cls is hg.DataCache:
                store.get_data(key)["a"] = 42
                self.assertEqual(store.get_data(key), {"a": 42})
                self.assertEqual(list(store), [(key, {"a": 42})])
                store.clear_all_data()
            else:
                store.__set__(key, 42)
                self.assertEqual(store.__get__(key, None), 42)
                self.assertEqual(list(store), [(key, 42)])
                store.clear()
            self.assertEqual(list(store), [])

    def test_plain_argument_helper_rejects_implicit_cache(self):
        @hg.argument_helper(hg.CptGridGraph)
        def shape_of(graph, shape=None):
            return shape

        graph = hg.get_4_adjacency_graph((2, 3))
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'data_cache'"):
            shape_of(graph, data_cache=None)
        self.assertEqual(shape_of(graph, shape=(3, 2)), (3, 2))
        self.assertEqual(shape_of(graph, shape=None), (2, 3))

    def test_no_backend_during_import_and_workflows(self):
        code = '''
import sys

def guard(frame, event, arg):
    if event == "call" and frame.f_code.co_name == "__init__":
        owner = frame.f_locals.get("self")
        if type(owner).__name__ in ("DataCache", "WeakKeyDictionary") and type(owner).__module__ == "higra.data_cache":
            raise AssertionError("metadata backend constructed")
sys.setprofile(guard)
import higra as hg
import numpy as np
assert not hasattr(hg, "__higra_global_cache")
g = hg.get_4_adjacency_graph((2, 3))
t, a = hg.bpt_canonical(g, np.arange(g.num_edges()))
hg.attribute_area(t)
hg.attribute_vertex_perimeter(g)
t.lowest_common_ancestor_preprocess()
hg.CptBinaryHierarchy.construct(t)
for obj in (g, t):
    assert "__higra_attributes__" not in vars(obj)
'''
        subprocess.run([sys.executable, "-Werror::FutureWarning", "-c", code],
                       cwd=Path(hg.__file__).resolve().parent.parent, check=True)

    def test_bound_owners_and_current_pickle_associations(self):
        import pickle

        optimized = hg.UndirectedGraphOptimizedDelete(2)
        hg.CptGridGraph.link(optimized, (1, 2))
        optimized.array = np.arange(2)
        self.assertEqual(hg.CptGridGraph.construct(optimized)["shape"], (1, 2))
        graph = hg.get_4_adjacency_graph((2, 3))
        tree, _ = hg.bpt_canonical(graph, np.arange(graph.num_edges(), dtype=np.float32))
        mst = hg.CptBinaryHierarchy.get_mst(tree)
        lca = tree.lowest_common_ancestor_preprocess()
        restored, g, m, edge_map, preprocessing = pickle.loads(pickle.dumps(
            (tree, graph, mst, tree.mst_edge_map, lca)))
        self.assertIs(restored.leaf_graph, g)
        self.assertIs(restored.mst, m)
        self.assertIs(m.base_graph, g)
        self.assertIs(m.mst_edge_map, edge_map)
        self.assertIs(restored.mst_edge_map, edge_map)
        self.assertIs(restored.lowest_common_ancestor_preprocess(), preprocessing)
        self.assertEqual(restored.category(), tree.category())
        self.assertEqual(edge_map.dtype, tree.mst_edge_map.dtype)
        self.assertEqual(hg.CptGridGraph.get_shape(g), (2, 3))
        self.assertTrue(hg.CptBinaryHierarchy.validate(restored))
        del tree.mst
        rebuilt = hg.CptBinaryHierarchy.get_mst(tree)
        self.assertIs(rebuilt, hg.CptBinaryHierarchy.get_mst(tree))
        self.assertIs(rebuilt.mst_edge_map, tree.mst_edge_map)
        self.assertIs(rebuilt.base_graph, graph)


if __name__ == '__main__':
    unittest.main()
