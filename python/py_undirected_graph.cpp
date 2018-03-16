//
// Created by user on 3/15/18.
//

#include "py_undirected_graph.hpp"
#include "pybind11/pybind11.h"
#include "undirected_graph.hpp"
#include "pybind11/stl.h"
#include <boost/graph/detail/edge.hpp>
#include <boost/iterator/transform_iterator.hpp>

namespace py = pybind11;

using graph_t = hg::undirected_graph;
using edge_t = typename boost::graph_traits<graph_t>::edge_descriptor;//boost::detail::edge_desc_impl<boost::undirected_tag, unsigned long>;
using vertex_t = typename boost::graph_traits<graph_t>::vertex_descriptor;
// for wrapping internal boost edge class to python tuple

using iterator_transform_function = std::function<py::tuple(edge_t)>;
using out_edge_iterator = boost::transform_iterator<iterator_transform_function, typename boost::graph_traits<graph_t>::out_edge_iterator>;
using in_edge_iterator = boost::transform_iterator<iterator_transform_function, typename boost::graph_traits<graph_t>::in_edge_iterator>;
using edge_iterator = boost::transform_iterator<iterator_transform_function, typename boost::graph_traits<graph_t>::edge_iterator>;

void py_init_undirected_graph(py::module &m) {


    py::class_<graph_t>(m, "UndirectedGraph")
            .def(py::init<const std::size_t>(), "Create a new graph with no edge.",
                 py::arg("numberOfVertices") = 0)
            .def("addEdge", [](hg::undirected_graph &g,
                               const hg::undirected_graph::vertex_descriptor source,
                               const hg::undirected_graph::vertex_descriptor target) {
                     boost::add_edge(source, target, g);
                 },
                 "Add an (undirected) edge between 'vertex1' and 'vertex2'",
                 py::arg("vertex1"),
                 py::arg("vertex2")
            ).def("addVertex", [](graph_t &g) {
                      return boost::add_vertex(g);
                  },
                  "Add a vertex to the graph, the index of the new vertex is returned"
            )
            .def("vertices", [](graph_t &g) {
                     auto it = boost::vertices(g);
                     return py::make_iterator(it.first, it.second);
                 },
                 "Iterator over all vertices of the graph.")
            .def("adjacentVertices", [](graph_t &g,
                                        const graph_t::vertex_descriptor v) {
                     auto it = boost::adjacent_vertices(v, g);
                     return py::make_iterator(it.first, it.second);
                 },
                 "Iterator over all vertices adjacent to the given vertex.",
                 py::arg("vertex"))
            .def("outEdges", [](graph_t &g,
                                const graph_t::vertex_descriptor v) {
                     auto it = boost::out_edges(v, g);
                     // wrapping out edge iterator to python friendly type
                     iterator_transform_function fun = [&g](edge_t e) -> py::tuple {
                         return py::make_tuple(boost::source(e, g), boost::target(e, g));
                     };
                     auto it1 = out_edge_iterator(it.first, fun);
                     auto it2 = out_edge_iterator(it.second, fun);
                     return py::make_iterator(it1, it2);

                 },
                 "Iterator over all out edges from 'vertex'. An out edge is a tuple '(vertex, adjacent_vertex)'.",
                 py::arg("vertex"))
            .def("inEdges", [](graph_t &g,
                               const graph_t::vertex_descriptor v) {
                     auto it = boost::in_edges(v, g);
                     // wrapping in edge iterator to python friendly type
                     iterator_transform_function fun = [&g](edge_t e) -> py::tuple {
                         return py::make_tuple(boost::source(e, g), boost::target(e, g));
                     };
                     auto it1 = in_edge_iterator(it.first, fun);
                     auto it2 = in_edge_iterator(it.second, fun);
                     return py::make_iterator(it1, it2);

                 },
                 "Iterator over all in edges from 'vertex'. An out edge is a tuple '(adjacent_vertex, vertex)'.",
                 py::arg("vertex"))
            .def("edges", [](graph_t &g) {
                     auto it = boost::edges(g);
                     // wrapping  edge iterator to python friendly type
                     iterator_transform_function fun = [&g](edge_t e) -> py::tuple {
                         return py::make_tuple(boost::source(e, g), boost::target(e, g));
                     };
                     auto it1 = edge_iterator(it.first, fun);
                     auto it2 = edge_iterator(it.second, fun);
                     return py::make_iterator(it1, it2);
                 },
                 "Iterator over all edges of the graph.")
            .def("degree", [](graph_t &g, vertex_t vertex) { return boost::degree(vertex, g); },
                 "Return the degree of the given vertex (same as inDegree and outDegree)",
                 py::arg("vertex"))
            .def("inDegree", [](graph_t &g, vertex_t vertex) { return boost::in_degree(vertex, g); },
                 "Return the in degree of the given vertex (same as degree and outDegree)",
                 py::arg("vertex"))
            .def("outDegree", [](graph_t &g, vertex_t vertex) { return boost::out_degree(vertex, g); },
                 "Return the out degree of the given vertex (same as degree and out degree)",
                 py::arg("vertex"))
            .def("numVertices", [](graph_t &g) { return boost::num_vertices(g); },
                 "Return the number of vertices in the graph")
            .def("numEdges", [](graph_t &g) { return boost::num_edges(g); },
                 "Return the number of edges in the graph");

    m.def("getTestUndirectedGraph", []() {
              hg::undirected_graph g(4);
              boost::add_edge(0, 1, g);
              boost::add_edge(1, 2, g);
              boost::add_edge(0, 2, g);
              return g;
          },
          "Returns a small undirected graph for testing purpose.");

}


