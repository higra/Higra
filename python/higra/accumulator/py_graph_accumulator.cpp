/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_graph_accumulator.hpp"
#include "../py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "higra/accumulator/graph_accumulator.hpp"

// @TODO Remove layout_type when xtensor solves the issue with iterators
template<typename T>
using pyarray = xt::pyarray<T, xt::layout_type::row_major>;

namespace py = pybind11;

using graph_t = hg::ugraph ;
using edge_t = graph_t::edge_descriptor;
using vertex_t = graph_t::vertex_descriptor;


template<typename graph_t>
struct def_accumulate_graph_edges {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_accumulate_graph_edges", [](const graph_t &graph, const pyarray<value_t> &input,
                                         hg::accumulators accumulator) {
                  switch (accumulator) {
                      case hg::accumulators::min:
                          return hg::accumulate_graph_edges(graph, input, hg::accumulator_min());
                          break;
                      case hg::accumulators::max:
                          return hg::accumulate_graph_edges(graph, input, hg::accumulator_max());
                          break;
                      case hg::accumulators::mean:
                          return hg::accumulate_graph_edges(graph, input, hg::accumulator_mean());
                          break;
                      case hg::accumulators::counter:
                          return hg::accumulate_graph_edges(graph, input, hg::accumulator_counter());
                          break;
                      case hg::accumulators::sum:
                          return hg::accumulate_graph_edges(graph, input, hg::accumulator_sum());
                          break;
                      case hg::accumulators::prod:
                          return hg::accumulate_graph_edges(graph, input, hg::accumulator_prod());
                          break;
                      case hg::accumulators::first:
                          return hg::accumulate_graph_edges(graph, input, hg::accumulator_first());
                          break;
                      case hg::accumulators::last:
                          return hg::accumulate_graph_edges(graph, input, hg::accumulator_last());
                          break;
                  }
                  throw std::runtime_error("Unknown accumulator.");
              },
              doc,
              py::arg("graph"),
              py::arg("input"),
              py::arg("accumulator"));
    }
};


template<typename graph_t>
struct def_accumulate_graph_vertices {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_accumulate_graph_vertices", [](const graph_t &graph, const pyarray<value_t> &input,
                                            hg::accumulators accumulator) {
                  switch (accumulator) {
                      case hg::accumulators::min:
                          return hg::accumulate_graph_vertices(graph, input, hg::accumulator_min());
                          break;
                      case hg::accumulators::max:
                          return hg::accumulate_graph_vertices(graph, input, hg::accumulator_max());
                          break;
                      case hg::accumulators::mean:
                          return hg::accumulate_graph_vertices(graph, input, hg::accumulator_mean());
                          break;
                      case hg::accumulators::counter:
                          return hg::accumulate_graph_vertices(graph, input, hg::accumulator_counter());
                          break;
                      case hg::accumulators::sum:
                          return hg::accumulate_graph_vertices(graph, input, hg::accumulator_sum());
                          break;
                      case hg::accumulators::prod:
                          return hg::accumulate_graph_vertices(graph, input, hg::accumulator_prod());
                          break;
                      case hg::accumulators::first:
                          return hg::accumulate_graph_vertices(graph, input, hg::accumulator_first());
                          break;
                      case hg::accumulators::last:
                          return hg::accumulate_graph_vertices(graph, input, hg::accumulator_last());
                          break;
                  }
                  throw std::runtime_error("Unknown accumulator.");
              },
              doc,
              py::arg("graph"),
              py::arg("input"),
              py::arg("accumulator"));
    }
};


void py_init_graph_accumulator(pybind11::module &m) {
    add_type_overloads<def_accumulate_graph_edges<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "");

    add_type_overloads<def_accumulate_graph_vertices<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "");
}