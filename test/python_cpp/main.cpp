/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <catch2/catch_test_macros.hpp>
#include "higra/python/py_array.hpp"
#include <pybind11/embed.h> // everything needed for embedding

namespace py = pybind11;
using namespace py::literals;
using namespace std;

namespace {
    struct holder {
        hg::array<double> data{1., 2., 3.};
    };
}

PYBIND11_EMBEDDED_MODULE(dummy, m
) {
    // dtype dispatch: the overload matching the input dtype is selected
    m.def("foo", [](hg::array_view<const float>) {
        return 42;
    });
    m.def("foo", [](hg::array_view<const double>) {
        return 43;
    });

    // NumPy input, in-place mutation and NumPy output
    m.def("increment_array", [](hg::array_view<double> array) {
        for (auto &v: array) {
            v += 1.0;
        }
        return hg::array<double>(array);
    });
    m.def("sum", [](hg::array_view<const double> array) {
        double s = 0;
        for (auto v: array) {
            s += v;
        }
        return s;
    });
    m.def("arange", [](hg::index_t n) {
        return hg::arange<double>(n);
    });

    // borrowed (read-only) view of data owned by a C++ object
    py::class_<holder>(m, "Holder")
            .def(py::init<>())
            .def("data", [](const holder &h) -> const hg::array<double> & { return h.data; },
                 py::return_value_policy::reference_internal);
}


TEST_CASE("basic python embedded interpreter", "[embedded]") {
    // A single interpreter session: NumPy cannot be imported again after the interpreter is finalized,
    // so the checks are not split into Catch2 sections (each section reruns the test case).
    py::initialize_interpreter();
    py::module sys = py::module::import("sys");

    // NumPy inputs, outputs, in-place mutation and borrowed views
    py::exec(R"(
import numpy as np
import dummy
assert dummy.foo(np.zeros(3, dtype=np.float32)) == 42
assert dummy.foo(np.zeros(3, dtype=np.float64)) == 43
assert dummy.foo(np.zeros(6, dtype=np.float32)[::2]) == 42

values = np.array([1., 2., 3.])
result = dummy.increment_array(values)
np.testing.assert_array_equal(values, [2., 3., 4.])
np.testing.assert_array_equal(result, values)

assert dummy.sum(np.arange(10.)[::3]) == 18.

r = dummy.arange(4)
np.testing.assert_array_equal(r, [0., 1., 2., 3.])
assert r.flags.writeable and r.flags.c_contiguous

h = dummy.Holder()
d = h.data()
np.testing.assert_array_equal(d, [1., 2., 3.])
assert not d.flags.writeable
assert d.base is h
)");

    // failed overload
    py::exec(R"(
import dummy
try:
    dummy.foo("bar")
    raise AssertionError("TypeError expected")
except TypeError:
    pass
)");
    REQUIRE(true);
    py::finalize_interpreter();
}
