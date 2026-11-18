############################################################################
# Copyright ESIEE Paris (2026)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

# Tests of the NumPy converters of higra/python/py_array.hpp, through the test-only extension module
# _hg_array_test built next to this file.

import gc
import importlib
import os
import sys
import unittest
import warnings
import weakref

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_expected = os.path.exists(os.path.join(_here, "_hg_array_test_expected"))


def _import_test_module():
    if _here not in sys.path:
        sys.path.insert(0, _here)
    try:
        return importlib.import_module("_hg_array_test")
    except ImportError:
        if _expected:
            # the build tree promises the module: a missing module is a failure, not a skip
            raise
        return None


mod = _import_test_module()

numeric_types = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64,
                 np.float32, np.float64]
cpp_names = ["int8_t", "uint8_t", "int16_t", "uint16_t", "int32_t", "uint32_t", "int64_t", "uint64_t",
             "float", "double"]


def collect():
    for _ in range(3):
        gc.collect()


@unittest.skipIf(mod is None, "the _hg_array_test module is only built in development builds")
class TestArrayInteropDispatch(unittest.TestCase):

    def test_exact_dtype_contiguous(self):
        for dt, name in zip(numeric_types, cpp_names):
            a = np.arange(12, dtype=dt).reshape(3, 4)
            self.assertEqual(mod.dtype_name(a), name, dt)

    def test_exact_dtype_non_contiguous(self):
        # A strided or Fortran input of a dtype other than the first registered one must not fall
        # through to the converting pass (which would pick int8).
        for dt, name in zip(numeric_types, cpp_names):
            base = np.arange(24, dtype=dt).reshape(4, 6)
            for a in (base[::2], base[:, ::3], base.T, np.asfortranarray(base), base[::-1],
                      base[:, 1]):
                self.assertEqual(mod.dtype_name(a), name, (dt, a.strides))

        self.assertEqual(mod.dtype_name(np.arange(10, dtype=np.float32)[::2]), "float")

    @staticmethod
    def expected_dispatch(a):
        # Reference dispatch rule (unchanged from earlier Higra versions): an ndarray whose
        # dtype has the kind and item size of a registered type selects that type, whatever its layout
        # and byte order; any other input (lists, scalars, bool, complex, float16...) is converted by
        # the first registered overload, int8_t.
        if isinstance(a, np.ndarray):
            for dt, name in zip(numeric_types, cpp_names):
                if a.dtype.kind == np.dtype(dt).kind and a.dtype.itemsize == np.dtype(dt).itemsize:
                    return name
        return "int8_t"

    def test_dispatch_reference_rule(self):
        base = np.arange(24).reshape(4, 6)
        inputs = [[1, 2, 3], [1.5, 2.5], [[1, 2], [3, 4]], 3, 2.5, True, [True, False],
                  np.float64(2), np.int32(2), np.array(2.0), np.array([True, False]),
                  np.zeros(3, dtype=np.complex64), np.zeros(3, dtype=np.float16),
                  np.zeros(3, dtype=">f8"), np.zeros(3, dtype=np.uintp), np.zeros(3, dtype=np.intp),
                  np.zeros(3, dtype=np.longlong), np.zeros(3, dtype=np.ulonglong),
                  np.zeros(3, dtype=np.byte), np.zeros(3, dtype=np.ubyte)]
        for dt in numeric_types:
            b = base.astype(dt)
            inputs += [b, b[::2], b.T, np.asfortranarray(b), b[:, 1], b[::-1], b.astype(b.dtype.newbyteorder())]
        with warnings.catch_warnings():
            # complex inputs are converted (with a warning)
            warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
            for a in inputs:
                self.assertEqual(mod.dtype_name(a), self.expected_dispatch(a), repr(a))

    def test_byte_swapped_input(self):
        for dt, name in zip(numeric_types, cpp_names):
            a = np.arange(6, dtype=np.dtype(dt).newbyteorder()).reshape(2, 3)
            self.assertEqual(mod.dtype_name(a), name, a.dtype)
            self.assertEqual(mod.dtype_name(a[:, ::2]), name, a.dtype)
            s, shape, address, _ = mod.view_info(a)
            self.assertEqual((s, shape), (15.0, (2, 3)))
            if a.dtype.itemsize > 1:  # byte order does not apply to 1-byte types
                self.assertNotEqual(address, a.ctypes.data)

    def test_non_contiguous_values(self):
        base = np.arange(24, dtype=np.float32).reshape(4, 6)
        for a in (base[::2], base.T, np.asfortranarray(base), base[::-1, 1:5:2]):
            s, shape, address, name = mod.view_info(a)
            self.assertEqual(name, "float")
            self.assertEqual(shape, a.shape)
            self.assertEqual(s, float(a.sum()))
            self.assertNotEqual(address, a.ctypes.data)

    def test_contiguous_input_is_not_copied(self):
        for dt in numeric_types:
            a = np.arange(6, dtype=dt).reshape(2, 3)
            s, shape, address, _ = mod.view_info(a)
            self.assertEqual(address, a.ctypes.data)
            self.assertEqual(shape, (2, 3))

    def test_unaligned_input_is_copied(self):
        buffer = bytearray(8 * 5 + 1)
        a = np.frombuffer(buffer, dtype=np.float64, count=5, offset=1)
        a[:] = [1, 2, 3, 4, 5]
        self.assertFalse(a.flags.aligned)
        s, shape, address, name = mod.view_info(a)
        self.assertEqual((s, shape, name), (15.0, (5,), "double"))
        self.assertNotEqual(address, a.ctypes.data)

    def test_lists_scalars_booleans(self):
        # Inputs that are not ndarrays only match in the converting pass: the first registered
        # overload (int8_t) wins.
        self.assertEqual(mod.dtype_name([1, 2, 3]), "int8_t")
        self.assertEqual(mod.dtype_name([1.5, 2.5]), "int8_t")
        self.assertEqual(mod.dtype_name(3), "int8_t")
        self.assertEqual(mod.dtype_name(2.5), "int8_t")
        self.assertEqual(mod.dtype_name(True), "int8_t")
        self.assertEqual(mod.dtype_name([True, False]), "int8_t")
        # a bool array has no exact overload: converting pass as well
        self.assertEqual(mod.dtype_name(np.array([True, False])), "int8_t")
        # NumPy scalars are not ndarrays either
        self.assertEqual(mod.dtype_name(np.float64(2)), "int8_t")
        # 0-d arrays are ndarrays
        self.assertEqual(mod.dtype_name(np.array(2.0)), "double")

        s, shape, _, name = mod.view_info_bool([True, False, True])
        self.assertEqual((s, shape, name), (2.0, (3,), "bool"))
        s, shape, _, name = mod.view_info_bool(np.array([[True], [False]]))
        self.assertEqual((s, shape, name), (1.0, (2, 1), "bool"))

    def test_conversion_values(self):
        s, shape, _, name = mod.view_info([[1, 2], [3, 4]])
        self.assertEqual((s, shape, name), (10.0, (2, 2), "int8_t"))
        s, shape, _, name = mod.view_info(7)
        self.assertEqual((s, shape, name), (7.0, (), "int8_t"))

    def test_noconvert(self):
        a = np.arange(10, dtype=np.float64)
        self.assertEqual(mod.view_info_noconvert(a[::3])[:2], (18.0, (4,)))
        with self.assertRaises(TypeError):
            mod.view_info_noconvert(np.arange(10, dtype=np.float32))
        with self.assertRaises(TypeError):
            mod.view_info_noconvert([1.0, 2.0])

    def test_invalid_conversion(self):
        with self.assertRaises(TypeError):
            mod.view_info("not an array")
        with self.assertRaises(TypeError):
            mod.view_info([[1, 2], [3]])

    def test_rank_above_8(self):
        a9 = np.zeros((1,) * 9, dtype=np.float32)
        a8 = np.zeros((1,) * 8, dtype=np.float32)
        self.assertEqual(mod.view_info(a8)[1], (1,) * 8)
        with self.assertRaisesRegex(ValueError, "more than 8 dimensions"):
            mod.dtype_name(a9)
        with self.assertRaisesRegex(ValueError, "more than 8 dimensions"):
            mod.view_info(a9[..., ::1].transpose())
        # converting pass
        with self.assertRaisesRegex(ValueError, "more than 8 dimensions"):
            mod.dtype_name(np.zeros((1,) * 9, dtype=np.complex64))
        with self.assertRaisesRegex(ValueError, "more than 8 dimensions"):
            mod.view_info(np.zeros((1,) * 9).tolist())
        # in-place and owning inputs
        with self.assertRaisesRegex(ValueError, "more than 8 dimensions"):
            mod.increment_float64(np.zeros((1,) * 9))
        with self.assertRaisesRegex(ValueError, "more than 8 dimensions"):
            mod.negate_copy(np.zeros((1,) * 9))

    def test_zero_size_and_0d_inputs(self):
        for shape in [(0,), (3, 0), (0, 2, 5)]:
            a = np.zeros(shape, dtype=np.int16)
            s, shape_seen, _, name = mod.view_info(a)
            self.assertEqual((s, shape_seen, name), (0.0, shape, "int16_t"))
        s, shape, _, name = mod.view_info(np.array(4.5))
        self.assertEqual((s, shape, name), (4.5, (), "double"))
        s, shape, _, name = mod.view_info(np.zeros((4, 0))[::2])
        self.assertEqual((s, shape, name), (0.0, (2, 0), "double"))


@unittest.skipIf(mod is None, "the _hg_array_test module is only built in development builds")
class TestArrayInteropInPlace(unittest.TestCase):

    def test_in_place_modification(self):
        for dt in numeric_types:
            a = np.arange(6, dtype=dt).reshape(2, 3)
            mod.increment(a)
            self.assertTrue(np.array_equal(a, np.arange(1, 7).reshape(2, 3)), dt)

    def test_in_place_0d_and_empty(self):
        a = np.array(1.5)
        mod.increment_float64(a)
        self.assertEqual(a, 2.5)
        e = np.zeros((0, 3))
        mod.increment_float64(e)
        self.assertEqual(e.shape, (0, 3))

    def test_in_place_rejections(self):
        a = np.arange(10, dtype=np.float64)
        with self.assertRaises(TypeError):
            mod.increment_float64(a.astype(np.float32))
        with self.assertRaises(TypeError):
            mod.increment_float64(a[::2])
        with self.assertRaises(TypeError):
            mod.increment_float64(np.asfortranarray(a.reshape(2, 5)))
        with self.assertRaises(TypeError):
            mod.increment_float64([1.0, 2.0])
        with self.assertRaises(TypeError):
            mod.increment_float64(2.0)
        r = a.copy()
        r.flags.writeable = False
        with self.assertRaises(TypeError):
            mod.increment_float64(r)
        buffer = bytearray(8 * 5 + 1)
        unaligned = np.frombuffer(buffer, dtype=np.float64, count=5, offset=1)
        with self.assertRaises(TypeError):
            mod.increment_float64(unaligned)
        with self.assertRaises(TypeError):
            mod.increment_float64(a.astype(">f8" if np.little_endian else "<f8"))
        self.assertTrue(np.array_equal(a, np.arange(10)))
        # overload set: a strided array of a registered dtype matches no overload
        with self.assertRaises(TypeError):
            mod.increment(np.arange(10, dtype=np.int32)[::2])
        with self.assertRaises(TypeError):
            mod.increment(np.array([True]))


@unittest.skipIf(mod is None, "the _hg_array_test module is only built in development builds")
class TestArrayInteropOwningInputs(unittest.TestCase):

    def test_copy_semantics(self):
        a = np.arange(5, dtype=np.float64)
        r = mod.negate_copy(a)
        self.assertTrue(np.array_equal(a, np.arange(5)))
        self.assertTrue(np.array_equal(r, -np.arange(5)))
        self.assertFalse(np.shares_memory(a, r))

        self.assertTrue(np.array_equal(mod.negate_copy([1, 2]), [-1, -2]))
        self.assertTrue(np.array_equal(mod.negate_copy(np.arange(6.0).reshape(2, 3).T),
                                       -np.arange(6.0).reshape(2, 3).T))
        self.assertEqual(mod.sum_const_ref(np.arange(5, dtype=np.int64)[::2]), 6)

    def test_kept_copy_is_independent(self):
        h = mod.Holder(3)
        a = np.array([5, 6, 7], dtype=np.int64)
        h.set(a)
        a[:] = 0
        self.assertEqual(h.sum(), 18)
        self.assertTrue(np.array_equal(h.data(), [5, 6, 7]))


@unittest.skipIf(mod is None, "the _hg_array_test module is only built in development builds")
class TestArrayInteropOutputs(unittest.TestCase):

    def test_returned_dtypes(self):
        for dt, name in zip(numeric_types, cpp_names):
            r = getattr(mod, "arange_" + name)(5)
            self.assertEqual(r.dtype, dt)
            self.assertTrue(np.array_equal(r, np.arange(5)))
        b = mod.arange_bool(4)
        self.assertEqual(b.dtype, np.bool_)
        self.assertTrue(np.array_equal(b, [True, False, True, False]))

    def test_zero_copy_return(self):
        r = mod.arange_double(10)
        self.assertIsNotNone(r.base)
        self.assertEqual(type(r.base).__name__, "PyCapsule")
        self.assertTrue(r.flags.writeable)
        self.assertTrue(r.flags.c_contiguous)
        self.assertTrue(r.flags.aligned)
        self.assertFalse(r.flags.owndata)
        self.assertEqual(r.ctypes.data % 64, 0)
        r[0] = 42
        self.assertEqual(r[0], 42)

    def test_returned_buffer_is_freed(self):
        collect()
        live0 = mod.counted_live()
        total0 = mod.counted_total()
        r = mod.counted_array(1000)
        self.assertEqual(mod.counted_live(), live0 + 1)
        self.assertEqual(mod.counted_total(), total0 + 1)
        self.assertTrue(np.array_equal(r, np.arange(1000)))
        v = r[10:20]
        del r
        collect()
        # still referenced through the slice
        self.assertEqual(mod.counted_live(), live0 + 1)
        self.assertTrue(np.array_equal(v, np.arange(10, 20)))
        del v
        collect()
        self.assertEqual(mod.counted_live(), live0)

        # many results at once
        arrays = [mod.counted_array(10) for _ in range(50)]
        self.assertEqual(mod.counted_live(), live0 + 50)
        del arrays
        collect()
        self.assertEqual(mod.counted_live(), live0)

    def test_zero_size_outputs(self):
        live0 = mod.counted_live()
        for shape in [(0,), (3, 0), (0, 0, 2)]:
            r = mod.zeros(list(shape))
            self.assertEqual(r.shape, shape)
            self.assertEqual(r.dtype, np.float32)
            self.assertTrue(r.flags.writeable)
        e = mod.counted_array(0)
        self.assertEqual(e.shape, (0,))
        self.assertEqual(mod.counted_live(), live0)

    def test_0d_output(self):
        r = mod.zeros([])
        self.assertEqual(r.shape, ())
        self.assertEqual(r.ndim, 0)
        self.assertEqual(r[()], 0)
        t = mod.make_tuple(2)
        self.assertEqual(t[2].shape, ())
        self.assertEqual(t[2][()], 2.5)

    def test_nested_returns(self):
        a, b = mod.make_pair(3)
        self.assertTrue(np.array_equal(a, [0, 1, 2]))
        self.assertEqual(a.dtype, np.int32)
        self.assertTrue(np.array_equal(b, np.ones((3, 2))))
        self.assertEqual(type(a.base).__name__, "PyCapsule")
        self.assertEqual(type(b.base).__name__, "PyCapsule")

        t = mod.make_tuple(4)
        self.assertIsInstance(t, tuple)
        self.assertEqual([x.dtype for x in t], [np.int64, np.uint8, np.float32])
        self.assertTrue(np.array_equal(t[0], np.arange(4)))

        lst = mod.make_list(3)
        self.assertIsInstance(lst, list)
        self.assertEqual([x.tolist() for x in lst], [[0], [1, 1], [2, 2, 2]])
        for x in lst:
            self.assertEqual(type(x.base).__name__, "PyCapsule")

    def test_views_returned_by_value_are_copied(self):
        a = mod.view_of_static()
        b = mod.view_of_static()
        self.assertTrue(np.array_equal(a, [1, 2, 3]))
        self.assertFalse(np.shares_memory(a, b))
        a[0] = 10
        self.assertEqual(mod.view_of_static()[0], 1)

        x = np.arange(4.0)
        y = mod.pass_through(x)
        self.assertFalse(np.shares_memory(x, y))
        self.assertTrue(np.array_equal(x, y))

    def test_numpy_dlpack(self):
        r = mod.arange_int64_t(5)
        d = np.from_dlpack(r)
        self.assertTrue(np.shares_memory(r, d))

    def test_torch_dlpack(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        r = mod.arange_float(5)
        t = torch.from_dlpack(r)
        self.assertEqual(t.data_ptr(), r.ctypes.data)
        t[0] = 10
        self.assertEqual(r[0], 10)


@unittest.skipIf(mod is None, "the _hg_array_test module is only built in development builds")
class TestArrayInteropBorrowed(unittest.TestCase):

    def test_borrowed_const_reference(self):
        h = mod.Holder(5)
        for v in (h.data(), h.data_view()):
            self.assertTrue(np.array_equal(v, np.arange(5)))
            self.assertFalse(v.flags.writeable)
            self.assertIs(v.base, h)
            with self.assertRaises(ValueError):
                v[0] = 1
            with self.assertRaises(ValueError):
                v.flags.writeable = True
        self.assertTrue(np.shares_memory(h.data(), h.data_view()))

    def test_borrowed_view_sees_cpp_changes(self):
        h = mod.Holder(3)
        v = h.data()
        m = h.mutable_data()
        self.assertTrue(m.flags.writeable)
        m[1] = 100
        self.assertEqual(v[1], 100)
        self.assertEqual(h.sum(), 102)
        mv = h.mutable_view()
        self.assertTrue(mv.flags.writeable)
        mv[2] = -2
        self.assertEqual(h.sum(), 98)

    def test_default_policy_copies(self):
        h = mod.Holder(3)
        c = h.data_copy()
        self.assertTrue(c.flags.writeable)
        self.assertFalse(np.shares_memory(c, h.data()))
        c[0] = 50
        self.assertEqual(h.sum(), 3)

    def test_borrowed_keeps_parent_alive(self):
        collect()
        alive0 = mod.Holder.alive()
        h = mod.Holder(4)
        self.assertEqual(mod.Holder.alive(), alive0 + 1)
        v = h.data()
        w = v[1:3]
        ref = weakref.ref(h)
        del h
        collect()
        self.assertIsNotNone(ref())
        self.assertEqual(mod.Holder.alive(), alive0 + 1)
        self.assertTrue(np.array_equal(v, np.arange(4)))
        del v
        collect()
        self.assertIsNotNone(ref())
        self.assertTrue(np.array_equal(w, [1, 2]))
        del w
        collect()
        self.assertIsNone(ref())
        self.assertEqual(mod.Holder.alive(), alive0)

    def test_borrowed_zero_size(self):
        h = mod.Holder(0)
        v = h.data()
        self.assertEqual(v.shape, (0,))
        self.assertFalse(v.flags.writeable)


@unittest.skipIf(mod is None, "the _hg_array_test module is only built in development builds")
class TestArrayInteropDtypes(unittest.TestCase):

    def test_dtype_agreement(self):
        for name, agrees, numpy_dtype, size in mod.dtype_agreement():
            self.assertTrue(agrees, (name, numpy_dtype, size))
            self.assertEqual(np.dtype(numpy_dtype).itemsize, size, name)


if __name__ == '__main__':
    unittest.main()
