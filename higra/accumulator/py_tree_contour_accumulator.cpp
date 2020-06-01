/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_tree_contour_accumulator.hpp"
#include "../py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "higra/accumulator/tree_contour_accumulator.hpp"
#include "common.hpp"

// @TODO Remove layout_type when xtensor solves the issue with iterators
template<typename T>
using pyarray = xt::pyarray<T, xt::layout_type::row_major>;

namespace py = pybind11;


template<typename graph_t, typename tree_t>
struct def_accumulate_on_contours {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_accumulate_on_contours",
              [](const graph_t &graph,
                 const tree_t &tree,
                 const pyarray<value_t> &vertex_data,
                 const pyarray<hg::index_t> &depth,
                 hg::accumulators accumulator) {
                  return dispatch_accumulator(
                          [&graph, &tree, &vertex_data, &depth](const auto &acc) {
                              return hg::accumulate_on_contours(graph, tree, vertex_data, depth, acc);
                          },
                          accumulator);
              },
              doc,
              py::arg("graph"),
              py::arg("tree"),
              py::arg("vertex_data"),
              py::arg("depth"),
              py::arg("accumulator"));
    }
};

void py_init_tree_contour_accumulator(pybind11::module &m) {
    xt::import_numpy();
    add_type_overloads<def_accumulate_on_contours<hg::ugraph, hg::tree>, HG_TEMPLATE_NUMERIC_TYPES>
            (m,
             "");
}