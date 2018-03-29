//
// Created by user on 3/15/18.
//

#include "pybind11/pybind11.h"
#include "py_undirected_graph.hpp"
#include "py_embedding.hpp"

#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"     // Numpy bindings

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


#ifdef VERSION_INFO
m.attr("__version__") = VERSION_INFO;
#else
m.attr("__version__") = "dev";
#endif
    xt::import_numpy();
    py_init_undirected_graph(m);
    py_init_embedding(m);
}