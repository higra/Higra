/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#define CATCH_CONFIG_MAIN

//#include "xtl/xmeta_utils.hpp"
#define FORCE_IMPORT_ARRAY

#include "catch2/catch.hpp"
#include <xtensor/xtensor.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>
#include <pybind11/embed.h> // everything needed for embedding

namespace py = pybind11;
using namespace py::literals;
//using namespace xt;
using namespace std;

PYBIND11_EMBEDDED_MODULE(dummy, m
) {
    xt::import_numpy();

    m.def("foo", [](
            xt::pytensor<float, 1> array
    ) {
        return 42;
    });
    m.def("foo", [](
            xt::pyarray<float> array
    ) {
        return 43;
    });
}


TEST_CASE("basic python embedded interpreter", "[embedded]") {
    py::initialize_interpreter();
    SECTION("example import") {
        py::module sys = py::module::import("sys");

        /*    py::scoped_interpreter guard{};
            auto locals = py::dict("name"_a = "World", "number"_a = 42);
            py::exec(R"(
            result = 42
        )", py::globals(), locals);
            BOOST_CHECK(locals["result"].cast<int>() == 42);*/
    }
    SECTION("failed overload") {
        py::module sys = py::module::import("sys");
        /*sys.attr("path") = py::make_tuple("", "/home/perretb/Higra/cmake-build-debug/python",
                "/home/perretb/anaconda3/lib/python36.zip",
                "/home/perretb/anaconda3/lib/python3.6",
                "/home/perretb/anaconda3/lib/python3.6/lib-dynload",
                "/home/perretb/.local/lib/python3.6/site-packages",
                "/home/perretb/anaconda3/lib/python3.6/site-packages");

        auto dummy_m = py::module::import("dummy");
        BOOST_CHECK_THROW(dummy_m.attr("foo")("bar").cast<int>(), py::error_already_set);
        */
    }
    SECTION("overload tensor array") {
        /* py::module sys = py::module::import("sys");
        sys.attr("path") = py::make_tuple("", "/home/perretb/Higra/cmake-build-debug/python",
                "/home/perretb/anaconda3/lib/python36.zip",
                "/home/perretb/anaconda3/lib/python3.6",
                "/home/perretb/anaconda3/lib/python3.6/lib-dynload",
                "/home/perretb/.local/lib/python3.6/site-packages",
                "/home/perretb/anaconda3/lib/python3.6/site-packages");

        auto dummy_m = py::module::import("dummy");

        xt::xtensor<float, 2> a{{1}};
        dummy_m.attr("foo")(a);*/
    }
    SECTION("higra sand box") {
        /*
        py::module sys = py::module::import("sys");
        sys.attr("path") = py::make_tuple("", "/home/perretb/Higra/cmake-build-debug/python",
                                          "/home/perretb/anaconda3/lib/python36.zip",
                                          "/home/perretb/anaconda3/lib/python3.6",
                                          "/home/perretb/anaconda3/lib/python3.6/lib-dynload",
                                          "/home/perretb/.local/lib/python3.6/site-packages",
                                          "/home/perretb/anaconda3/lib/python3.6/site-packages");

        auto higra_m = py::module::import("higra");
        */
        //auto res = higra_m.attr("read_graph_pink")("/home/perretb/2008_000009.graph");
    }
    py::finalize_interpreter();
}
