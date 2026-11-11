/***************************************************************************
* Copyright ESIEE Paris (2026)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

/**
 * Conversions between NumPy arrays and Higra arrays (``hg::array_view<const T>``,
 * ``hg::array_view<T>`` and ``hg::array<T>``) for pybind11 bindings.
 *
 * Include this header in every translation unit that binds a function taking or returning these
 * types: the converters are ``pybind11::detail::type_caster`` specializations and must be visible
 * wherever such a function is bound. The header can be included together with xtensor-python:
 * it only specializes converters for ``hg::`` types.
 *
 * Function parameters
 * -------------------
 *
 * ``array_view<const T>`` (read-only input):
 *  - without conversion (first pass of pybind11 overload resolution), accepts a NumPy array whose
 *    dtype has the kind and size of ``T`` (bool, signed or unsigned integer, floating point),
 *    whatever its memory layout and byte order. A non C-contiguous, unaligned or byte-swapped
 *    array is copied to C-contiguous native storage; otherwise the view shares the NumPy memory.
 *  - with conversion (second pass, or a function without ``noconvert``), additionally accepts
 *    anything NumPy can convert to an array of ``T``: arrays of other dtypes, lists, scalars.
 *  - 0-d arrays and scalars give a rank-0 view.
 *  - An array with more than ``hg::max_rank`` (8) dimensions raises ``ValueError``.
 *
 *  Because the first pass looks at the dtype only, a binding that registers one overload per
 *  element type (``HG_TEMPLATE_NUMERIC_TYPES``) dispatches a strided ``float32`` array to the
 *  ``float`` overload, never to a converting overload of another type. Inputs that are not NumPy
 *  arrays (lists, scalars, NumPy scalars) and arrays without an overload of their dtype go to the
 *  second pass, where the first registered overload that can convert them is selected. These are
 *  the rules of the xtensor-python ``pyarray`` converters.
 *
 *  The view is valid for the duration of the call only: it may point into a temporary copy.
 *
 * ``array_view<T>`` (in-place input):
 *  - accepts only a NumPy array with exactly the dtype of ``T`` (native byte order) that is
 *    C-contiguous, aligned and writeable; no conversion or copy is ever made, so writes are
 *    visible to the caller.
 *  - any other input makes the converter fail; if no other overload matches, pybind11 raises
 *    ``TypeError``. A Python wrapper that wants to support other inputs (lists, strided or
 *    read-only arrays...) must make a contiguous copy, call the binding, and write the result back.
 *
 * ``array<T>`` (by value or ``const array<T>&``): same rules as ``array_view<const T>``, then the
 * elements are copied into an array owned by C++, which may be kept after the call.
 *
 * Return values
 * -------------
 *
 *  - ``array<T>`` returned by value is moved to NumPy without copy: the NumPy array takes
 *    ownership of the buffer through a capsule (its ``base``) that releases it with the array's
 *    allocator. The result is writeable. A zero-size result is a new NumPy array.
 *  - ``const array<T>&`` and ``array_view<const T>`` returned with
 *    ``py::return_value_policy::reference_internal`` give a read-only NumPy array that shares the
 *    C++ memory; its ``base`` is the parent object (``self``), which is kept alive.
 *    ``array<T>&`` and ``array_view<T>`` returned with ``reference_internal`` give a writeable
 *    array that shares the C++ memory.
 *  - With any other policy, references and views are copied into a new NumPy array.
 *  - ``std::pair``, ``std::tuple`` and ``std::vector`` of arrays use pybind11's converters and
 *    apply the rules above element-wise (``pybind11/stl.h`` is needed for ``std::vector``).
 *
 * Ownership of returned buffers: the capsule keeps a pointer to the allocator of the array. The
 * CPU allocator is a function-local static of each extension module, so the buffer is freed by
 * the module that allocated it; extension modules are never unloaded by CPython.
 */

#pragma once

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "../structure/array/all.hpp"

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Hidden visibility, like pybind11's own namespace: the helpers hold pybind11 objects.
#if defined(__GNUG__) && !defined(_WIN32)
#define HG_PYTHON_INTERNAL_NAMESPACE python_internal __attribute__((visibility("hidden")))
#else
#define HG_PYTHON_INTERNAL_NAMESPACE python_internal
#endif

namespace hg {
    namespace HG_PYTHON_INTERNAL_NAMESPACE {

        namespace py = pybind11;

        /**
         * Raises ``ValueError`` if a NumPy array has more dimensions than ``hg::max_rank``.
         */
        inline void check_rank(py::ssize_t ndim) {
            if (ndim > static_cast<py::ssize_t>(max_rank)) {
                throw py::value_error("Arrays with more than " + std::to_string(max_rank) +
                                      " dimensions are not supported (got an array with " +
                                      std::to_string(ndim) + " dimensions).");
            }
        }

        inline py::ssize_t ndim_of(py::handle numpy_array) {
            return static_cast<py::ssize_t>(py::detail::array_proxy(numpy_array.ptr())->nd);
        }

        inline bool is_ndarray(py::handle h) {
            return py::detail::npy_api::get().PyArray_Check_(h.ptr());
        }

        inline bool has_flags(py::handle numpy_array, int flags) {
            return py::detail::check_flags(numpy_array.ptr(), flags);
        }

        /**
         * NumPy array of dtype ``T`` built from ``src`` by ``PyArray_FromAny`` with the given
         * requirement flags; an empty object if NumPy cannot convert ``src``.
         */
        template<class T>
        py::array from_any(py::handle src, int flags) {
            auto &api = py::detail::npy_api::get();
            PyObject *r = api.PyArray_FromAny_(src.ptr(), py::dtype::of<T>().release().ptr(), 0, 0,
                                               py::detail::npy_api::NPY_ARRAY_ENSUREARRAY_ | flags, nullptr);
            if (r == nullptr) {
                PyErr_Clear();
            }
            return py::reinterpret_steal<py::array>(r);
        }

        inline array_shape shape_of(const py::array &a) {
            const py::ssize_t *s = a.shape();
            return array_shape::from(std::vector<py::ssize_t>(s, s + a.ndim()));
        }

        inline std::vector<py::ssize_t> numpy_shape(const array_shape &shape) {
            return std::vector<py::ssize_t>(shape.begin(), shape.end());
        }

        /**
         * Higra dtype corresponding to a NumPy dtype; throws ``std::invalid_argument`` for dtypes
         * that have none (complex, object, structured...).
         */
        inline hg::dtype to_hg_dtype(const py::dtype &dt) {
            const auto bits = static_cast<uint8_t>(dt.itemsize() * 8);
            switch (dt.kind()) {
                case 'b':
                    return {dtype_code::bool_, bits};
                case 'i':
                    return {dtype_code::int_, bits};
                case 'u':
                    return {dtype_code::uint, bits};
                case 'f':
                    return {dtype_code::float_, bits};
                default:
                    throw std::invalid_argument(std::string("NumPy dtype kind '") + dt.kind() +
                                                "' has no Higra equivalent.");
            }
        }

        /**
         * True if a NumPy dtype describes the same values as ``T``: same kind (bool, signed,
         * unsigned, floating point) and item size, in any byte order.
         *
         * This is the dtype test of the first (no-convert) overload resolution pass. Like the
         * xtensor-python converters, it ignores the byte order, so that a byte-swapped array
         * selects the overload of its element type (and is then converted to native order).
         */
        template<class T>
        bool same_values(const py::dtype &dt) {
            constexpr hg::dtype expected = dtype_of<T>();
            if (dt.itemsize() * 8 != static_cast<py::ssize_t>(expected.bits)) {
                return false;
            }
            switch (dt.kind()) {
                case 'b':
                    return expected.code == dtype_code::bool_;
                case 'i':
                    return expected.code == dtype_code::int_;
                case 'u':
                    return expected.code == dtype_code::uint;
                case 'f':
                    return expected.code == dtype_code::float_;
                default:
                    return false;
            }
        }

        template<class T>
        bool is_ndarray_of(py::handle src) {
            if (!is_ndarray(src)) {
                return false;
            }
            auto descr = reinterpret_cast<PyObject *>(py::detail::array_proxy(src.ptr())->descr);
            return same_values<T>(py::reinterpret_borrow<py::dtype>(descr));
        }

        /**
         * True if pybind11 maps ``T`` to the NumPy dtype that ``hg::dtype_of<T>()`` describes.
         */
        template<class T>
        bool numpy_dtype_agrees() {
            return to_hg_dtype(py::dtype::of<T>()) == dtype_of<T>();
        }

        /**
         * Owner of a buffer released by ``array<T>::release()``, held by a capsule.
         */
        struct released_buffer_owner {
            void *data;
            std::size_t bytes;
            allocator *alloc;

            ~released_buffer_owner() {
                alloc->deallocate(data, bytes);
            }
        };

        /**
         * Moves an array to a new NumPy array without copying its elements.
         */
        template<class T>
        py::array to_numpy(array<T> &&a) {
            if (a.empty()) {
                return py::array(py::dtype::of<T>(), numpy_shape(a.shape()));
            }
            auto released = std::move(a).release();
            std::unique_ptr<released_buffer_owner> owner(
                    new released_buffer_owner{released.data, released.bytes, released.alloc});
            py::capsule base(owner.get(), [](void *p) {
                delete static_cast<released_buffer_owner *>(p);
            });
            owner.release(); // now owned by the capsule, also if the next line throws
            return py::array(py::dtype::of<T>(), numpy_shape(released.shape), released.data, base);
        }

        /**
         * NumPy array sharing the elements of ``v``, kept alive by ``parent``.
         *
         * If ``parent`` is null, the elements are copied instead, as pybind11 does.
         */
        template<class T>
        py::array borrow(array_view<const T> v, py::handle parent, bool writeable) {
            array_internal::assert_cpu_accessible(v.device());
            py::array r;
            if (v.empty()) {
                r = py::array(py::dtype::of<T>(), numpy_shape(v.shape()));
            } else {
                r = py::array(py::dtype::of<T>(), numpy_shape(v.shape()), v.data(), parent);
            }
            if (!writeable) {
                py::detail::array_proxy(r.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
            }
            return r;
        }

        /**
         * NumPy array holding a copy of the elements of ``v``.
         */
        template<class T>
        py::array copy_to_numpy(array_view<const T> v) {
            array_internal::assert_cpu_accessible(v.device());
            return to_numpy(array<T>(v));
        }

        /**
         * Conversion of a C++ view or array reference to Python according to the return value
         * policy (see the header documentation).
         */
        template<class T>
        py::handle cast_reference(array_view<const T> v, py::return_value_policy policy, py::handle parent,
                                  bool writeable) {
            if (policy == py::return_value_policy::reference_internal) {
                return borrow(v, parent, writeable).release();
            }
            return copy_to_numpy(v).release();
        }

        /**
         * Shared input conversion of ``array_view<const T>`` and ``array<T>``: keeps the NumPy
         * array that backs the view.
         */
        template<class T>
        struct const_view_loader {
            py::array storage;

            bool load(py::handle src, bool convert) {
                constexpr int required = py::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_ |
                                         py::detail::npy_api::NPY_ARRAY_ALIGNED_;
                if (!convert) {
                    // dtype only: never reject because of the layout or byte order (dispatch rule)
                    if (!is_ndarray_of<T>(src)) {
                        return false;
                    }
                    check_rank(ndim_of(src));
                    storage = from_any<T>(src, required);
                } else {
                    if (is_ndarray(src)) {
                        check_rank(ndim_of(src));
                    }
                    storage = from_any<T>(src, required | py::detail::npy_api::NPY_ARRAY_FORCECAST_);
                    if (storage) {
                        check_rank(storage.ndim());
                    }
                }
                return static_cast<bool>(storage);
            }

            array_view<const T> view() const {
                return array_view<const T>(static_cast<const T *>(storage.data()), shape_of(storage));
            }
        };
    }
}

#undef HG_PYTHON_INTERNAL_NAMESPACE

namespace pybind11 {
    namespace detail {

        template<class T>
        struct type_caster<hg::array_view<const T>> {
            PYBIND11_TYPE_CASTER(hg::array_view<const T>,
                                 const_name("numpy.ndarray[") + npy_format_descriptor<T>::name + const_name("]"));

            bool load(handle src, bool convert) {
                if (!m_loader.load(src, convert)) {
                    return false;
                }
                value = m_loader.view();
                return true;
            }

            static handle cast(hg::array_view<const T> src, return_value_policy policy, handle parent) {
                return hg::python_internal::cast_reference(src, policy, parent, false);
            }

        private:
            hg::python_internal::const_view_loader<T> m_loader;
        };

        template<class T>
        struct type_caster<hg::array_view<T>, enable_if_t<!std::is_const<T>::value>> {
            PYBIND11_TYPE_CASTER(hg::array_view<T>,
                                 const_name("numpy.ndarray[") + npy_format_descriptor<T>::name + const_name("]"));

            bool load(handle src, bool /*convert*/) {
                constexpr int required = npy_api::NPY_ARRAY_C_CONTIGUOUS_ | npy_api::NPY_ARRAY_ALIGNED_ |
                                         npy_api::NPY_ARRAY_WRITEABLE_;
                if (!array_t<T>::check_(src) || !hg::python_internal::has_flags(src, required)) {
                    return false;
                }
                auto a = reinterpret_borrow<array>(src);
                hg::python_internal::check_rank(a.ndim());
                m_storage = a;
                value = hg::array_view<T>(static_cast<T *>(a.mutable_data()), hg::python_internal::shape_of(a));
                return true;
            }

            static handle cast(hg::array_view<T> src, return_value_policy policy, handle parent) {
                return hg::python_internal::cast_reference(src.view(), policy, parent, true);
            }

        private:
            array m_storage;
        };

        template<class T>
        struct type_caster<hg::array<T>> {
            PYBIND11_TYPE_CASTER(hg::array<T>,
                                 const_name("numpy.ndarray[") + npy_format_descriptor<T>::name + const_name("]"));

            bool load(handle src, bool convert) {
                hg::python_internal::const_view_loader<T> loader;
                if (!loader.load(src, convert)) {
                    return false;
                }
                value = hg::array<T>(loader.view());
                return true;
            }

            static handle cast(hg::array<T> &&src, return_value_policy /*policy*/, handle /*parent*/) {
                return hg::python_internal::to_numpy(std::move(src)).release();
            }

            static handle cast(const hg::array<T> &src, return_value_policy policy, handle parent) {
                return hg::python_internal::cast_reference(src.view(), policy, parent, false);
            }

            static handle cast(hg::array<T> &src, return_value_policy policy, handle parent) {
                return hg::python_internal::cast_reference(src.view(), policy, parent, true);
            }
        };
    }
}
