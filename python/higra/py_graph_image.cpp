//
// Created by user on 4/2/18.
//

#include "py_graph_image.hpp"
#include "py_common_graph.hpp"
#include "higra/algo/graph_image.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "pybind11/functional.h"


namespace py = pybind11;

template<typename value_t>
void def_kalhimsky_2_contour(pybind11::module &m) {
    m.def("_khalimsky2Contour", [](const xt::pyarray<value_t> &khalimsky, bool extra_border) {
              return hg::khalimsky_2_contour2d(khalimsky, extra_border);
          },
          "Create a 4 adjacency edge-weighted graph from a contour image in the Khalimsky grid. Returns a tuple of three elements (graph, embedding, edge_weights).",
          py::arg("khalimsky"), py::arg("extraBorder") = false);
}

void py_init_graph_image(pybind11::module &m) {
    xt::import_numpy();

    m.def("_get4AdjacencyGraph", [](const std::vector<long> &v) {
              return hg::get_4_adjacency_graph(v);
          },
          "Create an explicit undirected 4 adjacency graph of the given shape.",
          py::arg("shape"));

    m.def("_get8AdjacencyGraph", [](const std::vector<long> &v) {
              return hg::get_8_adjacency_graph(v);
          },
          "Create an explicit undirected 8 adjacency graph of the given shape.",
          py::arg("shape"));

    m.def("_get4AdjacencyImplicitGraph", [](const std::vector<long> &v) {
              return hg::get_4_adjacency_implicit_graph(v);
          },
          "Create an explicit undirected 4 adjacency graph of the given shape (edges are not actually stored).",
          py::arg("shape"));

    m.def("_get8AdjacencyImplicitGraph", [](const std::vector<long> &v) {
              return hg::get_8_adjacency_implicit_graph(v);
          },
          "Create an explicit undirected 8 adjacency graph of the given shape (edges are not actually stored).",
          py::arg("shape"));

#define DEF(rawXKCD, dataXKCD, type) \
    m.def("_contour2Khalimsky", [](const hg::ugraph & graph, const std::vector<std::size_t> & shape, const xt::pyarray<type> & weights, bool add_extra_border){\
        hg::embedding_grid_2d embedding(shape); \
        return hg::contour2d_2_khalimsky(graph, embedding, weights, add_extra_border);},\
    "Create a contour image in the Khalimsky grid from a 4 adjacency edge-weighted graph (edge weights of type " HG_XSTR(type) ").",\
    py::arg("graph"),py::arg("shape"),py::arg("edgeWeights"),py::arg("addExtraBorder") = false);

    HG_FOREACH(DEF, HG_NUMERIC_TYPES);
#undef DEF

#define DEF(rawXKCD, dataXKCD, type) \
    def_kalhimsky_2_contour<type>(m);
    HG_FOREACH(DEF, HG_NUMERIC_TYPES);
#undef DEF
}

