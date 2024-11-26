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
#include "common.hpp"

namespace py_at_accumulator {
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
                      return dispatch_accumulator(
                              [&rag_map, &weights](const auto &acc) {
                                  return hg::accumulate_at(rag_map, weights, acc);
                              },
                              accumulator);
                  },
                  doc,
                  py::arg("indices"),
                  py::arg("weights"),
                  py::arg("accumulator")
            );
        }
    };


    void py_init_at_accumulator(pybind11::module &m) {
        //xt::import_numpy();
        add_type_overloads<def_at_accumulate, HG_TEMPLATE_NUMERIC_TYPES>
                (m, "");
    }
}