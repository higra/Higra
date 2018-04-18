//
// Created by user on 4/2/18.
//

#include "py_graph_weights.hpp"
#include "py_common_graph.hpp"
#include "higra/algo/graph_weights.hpp"
#include "xtensor-python/pyarray.hpp"
#include "pybind11/functional.h"


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


void py_init_graph_weights(pybind11::module &m) {
    xt::import_numpy();

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

}

