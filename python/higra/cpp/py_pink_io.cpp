//
// Created by user on 4/16/18.
//

#include "py_pink_io.hpp"
#include "py_common.hpp"
#include "higra/io/pink_graph_io.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

namespace py = pybind11;

template<typename graph_t>
void def_save(pybind11::module &m) {
    m.def("save_graph_pink", [](const std::string &filename,
                                const graph_t &graph,
                                const xt::pyarray<double> &vertex_values = {0},
                                const xt::pyarray<double> &edge_values = {0},
                                const std::vector<size_t> &shape = {}) {
              hg::save_pink_graph(filename, graph, vertex_values, edge_values, shape);
          },
          py::arg("filename"),
          py::arg("graph"),
          py::arg("vertex_weights") = xt::pyarray<double>(),
          py::arg("edge_weights") = xt::pyarray<double>(),
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