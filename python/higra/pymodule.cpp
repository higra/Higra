//
// Created by user on 3/15/18.
//

#include "pybind11/pybind11.h"
#include "py_undirected_graph.hpp"
#include "py_embedding.hpp"
#include "py_regular_graph.hpp"
#include "py_tree_graph.hpp"
#include "py_graph_weights.hpp"
#include "py_graph_image.hpp"
#include "py_lca_fast.hpp"
#include "py_hierarchy_core.hpp"
#include "py_pink_io.hpp"
#include "py_accumulators.hpp"

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
    py_init_graph_image(m);
    py_init_graph_weights(m);
    py_init_lca_fast(m);
    py_init_hierarchy_core(m);
    py_init_pink_io(m);
    py_init_accumulators(m);
}