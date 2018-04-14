//
// Created by user on 4/2/18.
//

#include "py_graph_misc.hpp"
#include "py_common_graph.hpp"
#include "xtensor-python/pyarray.hpp"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

namespace py = pybind11;

template<typename graph_t, typename value_t>
void weight_graph(pybind11::module &m) {
    m.def("weightGraph", [](const graph_t &graph, const xt::pyarray<value_t> &data, hg::weight_functions weight_f) {
              return hg::weight_graph(graph, data, weight_f);
          },
          "Compute the edge weights of a graph using source and target vertices values (of type " HG_XSTR(
                  value_t) ") and specified weighting function (see WeightFunction enumeration).",
          py::arg("explicitGraph"),
          py::arg("vertexWeights"),
          py::arg("weighFunction"));
}


void py_init_graph_misc(pybind11::module &m) {
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

    py::enum_<hg::weight_functions>(m, "WeightFunction")
            .value("mean", hg::weight_functions::mean)
            .value("min", hg::weight_functions::min)
            .value("max", hg::weight_functions::max)
            .value("L1", hg::weight_functions::L1)
            .value("L2", hg::weight_functions::L2)
            .value("L_infinity", hg::weight_functions::L_infinity)
            .value("L2_squared", hg::weight_functions::L2_squared);

#define DEF(rawXKCD, dataXKCD, type) \
        weight_graph<hg::ugraph, type>(m);
    HG_FOREACH(DEF, (int) (long) (float) (double));
#undef DEF

#define DEF(rawXKCD, dataXKCD, type) \
            weight_graph<hg::tree, type>(m);
    HG_FOREACH(DEF, (int) (long) (float) (double));
#undef DEF

    m.def("weightGraph", [](const hg::ugraph &graph, const std::function<double(std::size_t, std::size_t)> &fun) {
              return hg::weight_graph(graph, fun);
          },
          "Compute the edge weights of a graph with the given weighting function. The weighting function takes the "
                  "vertex index of the extremities of an edge and returns the weight of the edge",
          py::arg("explicitGraph"),
          py::arg("weightFunction"));

    m.def("weightGraph", [](const hg::tree &graph, const std::function<double(std::size_t, std::size_t)> &fun) {
              return hg::weight_graph(graph, fun);
          },
          "Compute the edge weights of a graph with the given weighting function. The weighting function takes the "
                  "vertex index of the extremities of an edge and returns the weight of the edge",
          py::arg("explicitGraph"),
          py::arg("weightFunction"));

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

