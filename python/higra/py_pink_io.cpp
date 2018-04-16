//
// Created by user on 4/16/18.
//

#include "py_pink_io.hpp"
#include "io/pink_graph_io.hpp"
#include "xtensor-python/pyarray.hpp"

namespace py = pybind11;

template<typename graph_t>
void def_save(pybind11::module &m) {
    m.def("saveGraphPink", [](const std::string &filename,
                              const graph_t &graph,
                              const xt::pyarray<double> &vertex_values = {0},
                              const xt::pyarray<double> &edge_values = {0},
                              const std::vector<std::size_t> &shape = {}) {
              hg::save_pink_graph(filename, graph, vertex_values, edge_values, shape);
          },
          py::arg("filename"),
          py::arg("graph"),
          py::arg("vertexWeights"),
          py::arg("edgeWeights"),
          py::arg("vertexWeights"));
}

void py_init_pink_io(pybind11::module &m) {
    xt::import_numpy();

    using class_t = hg::pink_graph<>;
    py::class_<class_t>(m, "__PinkGraph")
            .def_readwrite("graph", &class_t::graph)
            .def_readwrite("vertexWeights", &class_t::vertex_weights)
            .def_readwrite("edgeWeights", &class_t::edge_weights)
            .def_readwrite("shape", &class_t::shape);

    def_save<hg::ugraph>(m);

    m.def("readGraphPink", [](const std::string &filename) {
              return hg::read_pink_graph(filename);
          },
          py::arg("filename"));
}