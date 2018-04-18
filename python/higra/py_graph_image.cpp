//
// Created by user on 4/2/18.
//

#include "py_graph_image.hpp"
#include "py_common_graph.hpp"
#include "higra/algo/graph_image.hpp"
#include "xtensor-python/pyarray.hpp"
#include "pybind11/functional.h"


namespace py = pybind11;


void py_init_graph_image(pybind11::module &m) {
    xt::import_numpy();
    m.def("get4AdjacencyGraph", &hg::get_4_adjacency_graph,
          "Create an explicit undirected 4 adjacency graph of the given embedding.",
          py::arg("embedding2d"));
    m.def("get4AdjacencyGraph", [](const std::vector<long> &v) {
              return hg::get_4_adjacency_graph(v);
          },
          "Create an explicit undirected 4 adjacency graph of the given shape.",
          py::arg("shape"));

    m.def("get8AdjacencyGraph", &hg::get_8_adjacency_graph,
          "Create an explicit undirected 8 adjacency graph of the given embedding.",
          py::arg("embedding2d"));
    m.def("get8AdjacencyGraph", [](const std::vector<long> &v) {
              return hg::get_8_adjacency_graph(v);
          },
          "Create an explicit undirected 8 adjacency graph of the given shape.",
          py::arg("shape"));

    m.def("get4AdjacencyImplicitGraph", &hg::get_4_adjacency_implicit_graph,
          "Create an implicit 4 adjacency 2d graph of size given by the embedding (edges are not actually stored).",
          py::arg("embedding2d"));
    m.def("get4AdjacencyImplicitGraph", [](const std::vector<long> &v) {
              return hg::get_4_adjacency_implicit_graph(v);
          },
          "Create an explicit undirected 4 adjacency graph of the given shape (edges are not actually stored).",
          py::arg("shape"));

    m.def("get8AdjacencyImplicitGraph", &hg::get_8_adjacency_implicit_graph,
          "Create an implicit 8 adjacency 2d graph of size given by the embedding (edges are not actually stored).",
          py::arg("embedding2d"));
    m.def("get8AdjacencyImplicitGraph", [](const std::vector<long> &v) {
              return hg::get_8_adjacency_implicit_graph(v);
          },
          "Create an explicit undirected 8 adjacency graph of the given shape (edges are not actually stored).",
          py::arg("shape"));


#define DEF(rawXKCD, dataXKCD, type) \
    m.def("contour2Khalimsky", [](const hg::ugraph & graph, const hg::embedding_grid_2d & embedding, const xt::pyarray<type> & weights){\
        return hg::contour2d_2_khalimsky(graph, embedding, weights);},\
    "Create a contour image in the Khalimsky grid from a 4 adjacency edge-weighted graph (edge weights of type " HG_XSTR(type) ").",\
    py::arg("graph"),py::arg("embedding2d"),py::arg("edgeWeights"));

    HG_FOREACH(DEF, HG_NUMERIC_TYPES);
#undef DEF

#define DEF(rawXKCD, dataXKCD, type) \
    m.def("contour2Khalimsky", [](const hg::ugraph & graph, const std::vector<std::size_t> & shape, const xt::pyarray<type> & weights){\
        hg::embedding_grid_2d embedding(shape); \
        return hg::contour2d_2_khalimsky(graph, embedding, weights);},\
    "Create a contour image in the Khalimsky grid from a 4 adjacency edge-weighted graph (edge weights of type " HG_XSTR(type) ").",\
    py::arg("graph"),py::arg("shape"),py::arg("edgeWeights"));

    HG_FOREACH(DEF, HG_NUMERIC_TYPES);
#undef DEF
}

