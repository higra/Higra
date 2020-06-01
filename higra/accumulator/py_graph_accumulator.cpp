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
#include "common.hpp"

// @TODO Remove layout_type when xtensor solves the issue with iterators
template<typename T>
using pyarray = xt::pyarray<T, xt::layout_type::row_major>;

namespace py = pybind11;

using graph_t = hg::ugraph;
using edge_t = graph_t::edge_descriptor;
using vertex_t = graph_t::vertex_descriptor;


template<typename graph_t>
struct def_accumulate_graph_edges {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_accumulate_graph_edges", [](const graph_t &graph, const pyarray<value_t> &input,
                                            hg::accumulators accumulator) {
                  return dispatch_accumulator(
                          [&graph, &input](const auto &acc) {
                              return hg::accumulate_graph_edges(graph, input, acc);
                          },
                          accumulator);
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
                  return dispatch_accumulator(
                          [&graph, &input](const auto &acc) {
                              return hg::accumulate_graph_vertices(graph, input, acc);
                          },
                          accumulator);
              },
              doc,
              py::arg("graph"),
              py::arg("input"),
              py::arg("accumulator"));
    }
};


void py_init_graph_accumulator(pybind11::module &m) {
    xt::import_numpy();
    add_type_overloads<def_accumulate_graph_edges<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "");

    add_type_overloads<def_accumulate_graph_vertices<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "");
}