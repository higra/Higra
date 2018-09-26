/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_graph_weights.hpp"
#include "py_common_graph.hpp"
#include "higra/algo/graph_weights.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
template <typename T>
using pyarray = xt::pyarray<T, xt::layout_type::row_major>;
template <typename T, std::size_t N>
using pytensor = xt::pytensor<T, N, xt::layout_type::row_major>;
#include "pybind11/functional.h"


namespace py = pybind11;

template<typename graph_t>
struct def_weight_graph {
    template<typename type, typename C>
    static
    void def(C &m, const char *doc) {
        m.def("_weight_graph", [](const graph_t &graph, const pyarray<type> &data, hg::weight_functions weight_f) {
                  return hg::weight_graph(graph, data, weight_f);
              },
              doc,
              py::arg("explicit_graph"),
              py::arg("vertex_weights"),
              py::arg("weigh_function"));
    }
};

void py_init_graph_weights(pybind11::module &m) {
    xt::import_numpy();

    py::enum_<hg::weight_functions>(m, "WeightFunction")
            .value("mean", hg::weight_functions::mean)
            .value("min", hg::weight_functions::min)
            .value("max", hg::weight_functions::max)
            .value("L0", hg::weight_functions::L0)
            .value("L1", hg::weight_functions::L1)
            .value("L2", hg::weight_functions::L2)
            .value("L_infinity", hg::weight_functions::L_infinity)
            .value("L2_squared", hg::weight_functions::L2_squared);


    add_type_overloads<def_weight_graph<hg::ugraph>, HG_TEMPLATE_SNUMERIC_TYPES>
            (m,
             "Compute the edge weights of a graph using source and target vertices values"
                     " and specified weighting function (see WeightFunction enumeration)."
            );

    add_type_overloads<def_weight_graph<hg::tree>, HG_TEMPLATE_SNUMERIC_TYPES>
            (m,
             "Compute the edge weights of a graph using source and target vertices values"
                     " and specified weighting function (see WeightFunction enumeration)."
            );

    m.def("_weight_graph", [](const hg::ugraph &graph,
                              const std::function<double(hg::ugraph::vertex_descriptor, hg::ugraph::vertex_descriptor)> &fun) {
              return hg::weight_graph(graph, fun);
          },
          "Compute the edge weights of a graph with the given weighting function. The weighting function takes the "
                  "vertex index of the extremities of an edge and returns the weight of the edge",
          py::arg("explicit_graph"),
          py::arg("weight_function"));

    m.def("_weight_graph", [](const hg::tree &graph,
                              const std::function<double(hg::tree::vertex_descriptor, hg::tree::vertex_descriptor)> &fun) {
              return hg::weight_graph(graph, fun);
          },
          "Compute the edge weights of a graph with the given weighting function. The weighting function takes the "
                  "vertex index of the extremities of an edge and returns the weight of the edge",
          py::arg("explicit_graph"),
          py::arg("weight_function"));

}

