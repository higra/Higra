/***************************************************************************
* Copyright ESIEE Paris (2026)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

// Test-only extension module "_hg_array_test": exercises the NumPy converters of
// higra/python/py_array.hpp and the hg::bridge xtensor bridge through binding patterns used by
// Higra. It is built for the test suite only and never installed or packaged.
//
// The translation unit also includes xtensor-python and binds a pyarray function, as Higra
// bindings do while both array libraries coexist.

#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

#include "higra/python/py_array.hpp"
#include "higra/structure/array/xtensor_bridge.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "xtensor/core/xmath.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

    using hg::array;
    using hg::array_shape;
    using hg::array_view;
    using hg::index_t;

    template<class T>
    const char *type_name();

#define HG_TEST_TYPE_NAME(T) template<> const char *type_name<T>() { return #T; }

    HG_TEST_TYPE_NAME(int8_t)
    HG_TEST_TYPE_NAME(uint8_t)
    HG_TEST_TYPE_NAME(int16_t)
    HG_TEST_TYPE_NAME(uint16_t)
    HG_TEST_TYPE_NAME(int32_t)
    HG_TEST_TYPE_NAME(uint32_t)
    HG_TEST_TYPE_NAME(int64_t)
    HG_TEST_TYPE_NAME(uint64_t)
    HG_TEST_TYPE_NAME(float)
    HG_TEST_TYPE_NAME(double)
    HG_TEST_TYPE_NAME(bool)

#undef HG_TEST_TYPE_NAME

    // Allocator counting live buffers, to check that NumPy releases returned buffers.
    struct counting_allocator final : hg::allocator {
        index_t live = 0;
        index_t total = 0;

        void *allocate(std::size_t bytes) override {
            ++live;
            ++total;
            return hg::cpu_allocator().allocate(bytes);
        }

        void deallocate(void *p, std::size_t bytes) noexcept override {
            --live;
            hg::cpu_allocator().deallocate(p, bytes);
        }

        hg::device device() const noexcept override {
            return hg::cpu_device;
        }
    };

    counting_allocator &counter() {
        static counting_allocator instance;
        return instance;
    }

    // Object owning an array, returned to Python by reference.
    struct holder {
        static index_t &alive() {
            static index_t n = 0;
            return n;
        }

        array<int64_t> values;

        explicit holder(index_t n) : values(array_shape(n)) {
            for (index_t i = 0; i < n; ++i) {
                values[i] = i;
            }
            ++alive();
        }

        holder(const holder &) = delete;

        ~holder() {
            --alive();
        }
    };

    template<class T>
    py::tuple view_info(array_view<const T> v) {
        double s = 0;
        for (const auto &x: v) {
            s += static_cast<double>(x);
        }
        py::tuple shape(v.dimension());
        for (index_t k = 0; k < v.dimension(); ++k) {
            shape[k] = v.shape(k);
        }
        return py::make_tuple(s, shape, reinterpret_cast<std::uintptr_t>(v.data()), type_name<T>());
    }

    struct def_dtype_name {
        template<class T>
        static void def(py::module_ &m) {
            m.def("dtype_name", [](array_view<const T>) { return std::string(type_name<T>()); },
                  py::arg("array"));
        }
    };

    // Same overload set with the xtensor-python converter, as reference for the dispatch rules.
    struct def_pyarray_dtype_name {
        template<class T>
        static void def(py::module_ &m) {
            m.def("pyarray_dtype_name", [](const xt::pyarray<T> &) { return std::string(type_name<T>()); },
                  py::arg("array"));
        }
    };

    struct def_view_info {
        template<class T>
        static void def(py::module_ &m) {
            m.def("view_info", &view_info<T>, py::arg("array"));
        }
    };

    struct def_increment {
        template<class T>
        static void def(py::module_ &m) {
            m.def("increment", [](array_view<T> a) {
                for (auto &x: a) {
                    x = static_cast<T>(x + 1);
                }
            }, py::arg("array"));
        }
    };

    struct def_arange {
        template<class T>
        static void def(py::module_ &m) {
            m.def((std::string("arange_") + type_name<T>()).c_str(), [](index_t n) {
                return hg::arange<T>(static_cast<T>(n));
            }, py::arg("n"));
        }
    };

    template<class F, class... T>
    void def_for_types(py::module_ &m) {
        (F::template def<T>(m), ...);
    }

    array_shape to_shape(const std::vector<index_t> &dims) {
        return array_shape::from(dims);
    }

    template<class T>
    py::tuple dtype_agreement_entry(const char *name) {
        return py::make_tuple(name, hg::python_internal::numpy_dtype_agrees<T>(), py::dtype::of<T>().attr("str"),
                              sizeof(T));
    }
}

PYBIND11_MODULE(_hg_array_test, m) {
    xt::import_numpy();
    m.doc() = "Test-only module for the Higra NumPy converters and the xtensor bridge.";

    // --- inputs: read-only views, dispatch over the numeric types (registration order of
    // HG_TEMPLATE_NUMERIC_TYPES) ---
    def_for_types<def_dtype_name, HG_TEMPLATE_NUMERIC_TYPES>(m);
    def_for_types<def_pyarray_dtype_name, HG_TEMPLATE_NUMERIC_TYPES>(m);
    def_for_types<def_view_info, HG_TEMPLATE_NUMERIC_TYPES>(m);
    m.def("view_info_noconvert", &view_info<double>, py::arg("array").noconvert());
    m.def("view_info_bool", &view_info<bool>, py::arg("array"));

    // --- in-place inputs ---
    def_for_types<def_increment, HG_TEMPLATE_NUMERIC_TYPES>(m);
    m.def("increment_float64", [](array_view<double> a) {
        for (auto &x: a) {
            x += 1;
        }
    }, py::arg("array"));

    // --- owning inputs: copy semantics ---
    m.def("negate_copy", [](array<double> a) {
        for (auto &x: a) {
            x = -x;
        }
        return a;
    }, py::arg("array"));
    m.def("sum_const_ref", [](const array<int64_t> &a) { return hg::sum(a); }, py::arg("array"));

    // --- outputs ---
    def_for_types<def_arange, HG_TEMPLATE_NUMERIC_TYPES>(m);
    m.def("arange_bool", [](index_t n) {
        auto r = array<bool>(array_shape(n));
        for (index_t i = 0; i < n; ++i) {
            r[i] = i % 2 == 0;
        }
        return r;
    }, py::arg("n"));
    m.def("zeros", [](const std::vector<index_t> &shape) { return hg::zeros<float>(to_shape(shape)); },
          py::arg("shape"));
    m.def("counted_array", [](index_t n) {
        auto r = array<double>(array_shape(n), counter());
        for (index_t i = 0; i < n; ++i) {
            r[i] = static_cast<double>(i);
        }
        return r;
    }, py::arg("n"));
    m.def("counted_live", []() { return counter().live; });
    m.def("counted_total", []() { return counter().total; });
    m.def("make_pair", [](index_t n) {
        return std::make_pair(hg::arange<int32_t>(static_cast<int32_t>(n)), hg::ones<double>({n, 2}));
    }, py::arg("n"));
    m.def("make_tuple", [](index_t n) {
        return std::make_tuple(hg::arange<int64_t>(n), hg::zeros<uint8_t>({n}), array<float>(array_shape{}, 2.5f));
    }, py::arg("n"));
    m.def("make_list", [](index_t n) {
        std::vector<array<int64_t>> r;
        for (index_t i = 0; i < n; ++i) {
            r.push_back(hg::full<int64_t>({i + 1}, i));
        }
        return r;
    }, py::arg("n"));
    m.def("view_of_static", []() {
        static array<int64_t> data{1, 2, 3};
        return data.view();
    }, "Default policy: the returned view is copied.");
    m.def("pass_through", [](array_view<const double> v) { return v; }, py::arg("array"),
          "Default policy: returning an input view copies it.");

    py::class_<holder>(m, "Holder")
            .def(py::init<index_t>(), py::arg("n"))
            .def("data", [](const holder &h) -> const array<int64_t> & { return h.values; },
                 py::return_value_policy::reference_internal)
            .def("data_view", [](const holder &h) { return h.values.view(); },
                 py::return_value_policy::reference_internal)
            .def("data_copy", [](const holder &h) -> const array<int64_t> & { return h.values; })
            .def("mutable_data", [](holder &h) -> array<int64_t> & { return h.values; },
                 py::return_value_policy::reference_internal)
            .def("mutable_view", [](holder &h) { return array_view<int64_t>(h.values); },
                 py::return_value_policy::reference_internal)
            .def("set", [](holder &h, array<int64_t> a) { h.values = std::move(a); }, py::arg("array"))
            .def("sum", [](const holder &h) { return hg::sum(h.values); })
            .def_static("alive", []() { return holder::alive(); });

    // --- dtype mapping ---
    m.def("dtype_agreement", []() {
        return std::vector<py::tuple>{
                dtype_agreement_entry<bool>("bool"),
                dtype_agreement_entry<char>("char"),
                dtype_agreement_entry<signed char>("signed char"),
                dtype_agreement_entry<unsigned char>("unsigned char"),
                dtype_agreement_entry<short>("short"),
                dtype_agreement_entry<unsigned short>("unsigned short"),
                dtype_agreement_entry<int>("int"),
                dtype_agreement_entry<unsigned int>("unsigned int"),
                dtype_agreement_entry<long>("long"),
                dtype_agreement_entry<unsigned long>("unsigned long"),
                dtype_agreement_entry<long long>("long long"),
                dtype_agreement_entry<unsigned long long>("unsigned long long"),
                dtype_agreement_entry<std::size_t>("size_t"),
                dtype_agreement_entry<std::ptrdiff_t>("ptrdiff_t"),
                dtype_agreement_entry<hg::index_t>("hg::index_t"),
                dtype_agreement_entry<int8_t>("int8_t"),
                dtype_agreement_entry<uint8_t>("uint8_t"),
                dtype_agreement_entry<int16_t>("int16_t"),
                dtype_agreement_entry<uint16_t>("uint16_t"),
                dtype_agreement_entry<int32_t>("int32_t"),
                dtype_agreement_entry<uint32_t>("uint32_t"),
                dtype_agreement_entry<int64_t>("int64_t"),
                dtype_agreement_entry<uint64_t>("uint64_t"),
                dtype_agreement_entry<float>("float"),
                dtype_agreement_entry<double>("double")};
    });

    // --- xtensor bridge ---
    m.def("bridge_sum", [](array_view<const double> v) {
        return xt::sum(hg::bridge::as_xt(v))();
    }, py::arg("array"));
    m.def("bridge_sum_2d", [](array_view<const double> v) {
        auto x = hg::bridge::as_xt<2>(v);
        return py::make_tuple(xt::sum(x)(), reinterpret_cast<std::uintptr_t>(x.data()));
    }, py::arg("array"));
    m.def("bridge_increment", [](array_view<double> v) {
        auto x = hg::bridge::as_xt(v);
        x += 1;
    }, py::arg("array"));
    m.def("bridge_view_of_adaptor", [](array_view<const double> v) {
        auto h = hg::bridge::as_view(hg::bridge::as_xt(v));
        return py::make_tuple(array<double>(h.view), h.is_zero_copy(),
                              reinterpret_cast<std::uintptr_t>(h.view.data()) ==
                              reinterpret_cast<std::uintptr_t>(v.data()));
    }, py::arg("array"));
    m.def("bridge_view_of_lazy", [](array_view<const double> v) {
        auto h = hg::bridge::as_view(hg::bridge::as_xt(v) * 2 + 1);
        return py::make_tuple(array<double>(h.view), h.is_zero_copy());
    }, py::arg("array"));
    m.def("bridge_to_xt_round_trip", [](array_view<const double> v) {
        xt::xarray<double> x = hg::bridge::to_xt(v);
        x += 1;
        return hg::bridge::to_array(x);
    }, py::arg("array"));

    // --- coexistence with xtensor-python in one overload set ---
    m.def("mixed", [](const xt::pyarray<double> &a) { return std::string("pyarray"); }, py::arg("array"));
    m.def("mixed", [](array_view<const int32_t> a) { return std::string("array_view"); }, py::arg("array"));
    m.def("pyarray_echo", [](const xt::pyarray<double> &a) {
        xt::pyarray<double> r = a + 1;
        return r;
    }, py::arg("array"));
}
