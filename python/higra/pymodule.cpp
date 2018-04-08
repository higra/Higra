//
// Created by user on 3/15/18.
//

#include "pybind11/pybind11.h"
#include "py_undirected_graph.hpp"
#include "py_embedding.hpp"
#include "py_regular_graph.hpp"
#include "py_tree_graph.hpp"
#include "py_graph_misc.hpp"

#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"     // Numpy bindings

PYBIND11_MODULE(higram, m
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
    py_init_regular_graph(m);
    py_init_tree_graph(m);
    py_init_graph_misc(m);
}