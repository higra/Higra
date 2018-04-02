//
// Created by user on 4/1/18.
//

#include "py_regular_graph.hpp"
#include "py_common_graph.hpp"
#include "xtensor/xgenerator.hpp"

#include <vector>

namespace py = pybind11;
using embedding_t = hg::embedding_grid;
using graph_t = hg::regular_grid_graph;
using edge_t = typename boost::graph_traits<graph_t>::edge_descriptor;
using vertex_t = typename boost::graph_traits<graph_t>::vertex_descriptor;

void py_init_regular_graph(pybind11::module &m) {
    auto c = py::class_<graph_t>(m, "RegularGraph");

    c.def(py::init([](const embedding_t &e, const std::vector<std::vector<long>> &pl) {
                       std::vector<xt::xarray<long>> points;
                       auto dim = e.dimension();
                       for (const auto &v : pl) {
                           xt::xarray<long> pt = xt::zeros<long>({dim});
                           for (std::size_t i = 0; i < dim; i++)
                               pt(i) = v[i];
                           points.push_back(pt);
                       }
                       return graph_t(e, points);
                   }
          ),
          "Create a regular implicit graph from given embedding and neighbouring.",
          py::arg("embedding"),
          py::arg("neighbour_list"));

    c.def_static("get4AdjacencyGraph",
                 [](const embedding_t &e) { return hg::get_4_adjacency_implicit_graph(e); },
                 "Create a 4 adjacency 2d graph of size given by the embedding.",
                 py::arg("embedding"));

    c.def_static("get8AdjacencyGraph",
                 [](const embedding_t &e) { return hg::get_8_adjacency_implicit_graph(e); },
                 "Create a 8 adjacency 2d graph of size given by the embedding.",
                 py::arg("embedding"));


    add_incidence_graph_concept<graph_t, decltype(c)>(c);
    add_bidirectionnal_graph_concept<graph_t, decltype(c)>(c);
    add_adjacency_graph_concept<graph_t, decltype(c)>(c);
    add_vertex_list_graph_concept<graph_t, decltype(c)>(c);
}