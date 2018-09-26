/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_pink_io.hpp"
#include "py_common.hpp"
#include "higra/io/pink_graph_io.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

template<typename T>
using pyarray = xt::pyarray<T, xt::layout_type::row_major>;
template<typename T, std::size_t N>
using pytensor = xt::pytensor<T, N, xt::layout_type::row_major>;

namespace py = pybind11;

template<typename graph_t>
void def_save(pybind11::module &m) {
    m.def("save_graph_pink", [](const std::string &filename,
                                const graph_t &graph,
                                const pytensor<double, 1> &vertex_values = {0},
                                const pytensor<double, 1> &edge_values = {0},
                                const std::vector<size_t> &shape = {}) {
              hg::save_pink_graph(filename, graph, vertex_values, edge_values, shape);
          },
          py::arg("filename"),
          py::arg("graph"),
          py::arg("vertex_weights") = pytensor<double, 1>(),
          py::arg("edge_weights") = pytensor<double, 1>(),
          py::arg("shape") = std::vector<size_t>());
}

void py_init_pink_io(pybind11::module &m) {
    xt::import_numpy();

    def_save<hg::ugraph>(m);

    m.def("_read_graph_pink", [](const std::string &filename) {
              auto res = hg::read_pink_graph(filename);
              return std::make_tuple(std::move(res.graph), std::move(res.vertex_weights), std::move(res.edge_weights),
                                     std::move(res.shape));
          },
          py::arg("filename"));
}