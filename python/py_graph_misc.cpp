//
// Created by user on 4/2/18.
//

#include "py_graph_misc.hpp"
#include "py_common_graph.hpp"

#include "py_embedding.hpp"


namespace py = pybind11;


void py_init_graph_misc(pybind11::module &m) {
    m.def("get4AdjacencyGraph", &hg::get_4_adjacency_graph,
          "Create an explicit undirected 4 adjacency graph of the given dimensions.",
          py::arg("embedding2d"));

    m.def("get8AdjacencyGraph", &hg::get_8_adjacency_graph,
          "Create an explicit undirected 8 adjacency graph of the given dimensions.",
          py::arg("embedding2d"));

    m.def("get4AdjacencyImplicitGraph", &hg::get_4_adjacency_implicit_graph,
          "Create an implicit 4 adjacency 2d graph of size given by the embedding (edges are not actually stored).",
          py::arg("embedding2d"));

    m.def("get8AdjacencyImplicitGraph", &hg::get_8_adjacency_implicit_graph,
          "Create an implicit 8 adjacency 2d graph of size given by the embedding (edges are not actually stored).",
          py::arg("embedding2d"));
}