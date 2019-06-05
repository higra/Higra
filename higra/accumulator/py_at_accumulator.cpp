/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_at_accumulator.hpp"
#include "../py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "higra/accumulator/accumulator.hpp"
#include "higra/accumulator/at_accumulator.hpp"



namespace py = pybind11;
template<typename T>
using pyarray = xt::pyarray<T, xt::layout_type::row_major>;

namespace py = pybind11;

struct def_at_accumulate {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_accumulate_at",
              [](const pyarray<hg::index_t> &rag_map,
                 const pyarray<value_t> &weights,
                 hg::accumulators accumulator) {
                  switch (accumulator) {
                      case hg::accumulators::min:
                          return hg::accumulate_at(rag_map, weights, hg::accumulator_min());
                          break;
                      case hg::accumulators::max:
                          return hg::accumulate_at(rag_map, weights, hg::accumulator_max());
                          break;
                      case hg::accumulators::mean:
                          return hg::accumulate_at(rag_map, weights, hg::accumulator_mean());
                          break;
                      case hg::accumulators::counter:
                          return hg::accumulate_at(rag_map, weights, hg::accumulator_counter());
                          break;
                      case hg::accumulators::sum:
                          return hg::accumulate_at(rag_map, weights, hg::accumulator_sum());
                          break;
                      case hg::accumulators::prod:
                          return hg::accumulate_at(rag_map, weights, hg::accumulator_prod());
                          break;
                      case hg::accumulators::first:
                          return hg::accumulate_at(rag_map, weights, hg::accumulator_first());
                          break;
                      case hg::accumulators::last:
                          return hg::accumulate_at(rag_map, weights, hg::accumulator_last());
                          break;
                  }
                  throw std::runtime_error("Unknown accumulator.");
              },
              doc,
              py::arg("indices"),
              py::arg("weights"),
              py::arg("accumulator")
        );
    }
};


void py_init_at_accumulator(pybind11::module &m) {
    add_type_overloads<def_at_accumulate, HG_TEMPLATE_NUMERIC_TYPES>
            (m, "");
}