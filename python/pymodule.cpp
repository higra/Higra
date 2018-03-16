//
// Created by user on 3/15/18.
//

#include "pybind11/pybind11.h"
#include "py_undirected_graph.hpp"

#define FORCE_IMPORT_ARRAY


PYBIND11_MODULE(higra, m
) {
m.

doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";
/*
m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");
*/

#ifdef VERSION_INFO
m.attr("__version__") = VERSION_INFO;
#else
m.attr("__version__") = "dev";
#endif

py_init_undirected_graph(m);

}