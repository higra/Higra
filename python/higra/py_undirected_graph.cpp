//
// Created by user on 3/15/18.
//


#include "py_common_graph.hpp"
#include "xtensor-python/pyarray.hpp"
#include "pybind11/stl.h"

namespace py = pybind11;

using graph_t = hg::undirected_graph<>;
using edge_t = typename boost::graph_traits<graph_t>::edge_descriptor;
using vertex_t = typename boost::graph_traits<graph_t>::vertex_descriptor;

void py_init_undirected_graph(py::module &m) {


    auto c = py::class_<graph_t>(m, "UndirectedGraph");

    c.def(py::init<const std::size_t>(), "Create a new graph with no edge.",
          py::arg("numberOfVertices") = 0);


    add_incidence_graph_concept<graph_t, decltype(c)>(c);
    add_bidirectionnal_graph_concept<graph_t, decltype(c)>(c);
    add_adjacency_graph_concept<graph_t, decltype(c)>(c);
    add_vertex_list_graph_concept<graph_t, decltype(c)>(c);
    add_edge_list_graph_concept<graph_t, decltype(c)>(c);
    add_edge_index_graph_concept<graph_t, decltype(c)>(c);

    c.def("addEdge", [](graph_t &g,
                        const vertex_t source,
                        const vertex_t target) {
              hg::add_edge(source, target, g);
          },
          "Add an (undirected) edge between 'vertex1' and 'vertex2'",
          py::arg("vertex1"),
          py::arg("vertex2"));
    c.def("addVertex", [](graph_t &g) {
              return boost::add_vertex(g);
          },
          "Add a vertex to the graph, the index of the new vertex is returned");


    m.def("getTestUndirectedGraph", []() {
              hg::undirected_graph<> g(4);
              hg::add_edge(0, 1, g);
              hg::add_edge(1, 2, g);
              hg::add_edge(0, 2, g);
              return g;
          },
          "Returns a small undirected graph for testing purpose.");

}


