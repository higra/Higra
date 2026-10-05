/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_undirected_graph.hpp"
#include "py_common_graph.hpp"
#include <pybind11/numpy.h>
#include <memory>

namespace py_undirected_graph {
    using namespace py_common_graph;
    namespace py = pybind11;

    // Only the capsule owns Python references; native storage cannot form a cycle.
    template<typename graph_t>
    struct endpoint_array_owner {
        typename graph_t::edge_storage_handle storage;
        py::object graph;
        const hg::index_t empty_endpoint = 0;
    };

    template<typename graph_t>
    py::array endpoint_array(py::object graph, bool source) {
        const auto &g = graph.cast<const graph_t &>();
        auto owner = std::make_unique<endpoint_array_owner<graph_t>>();
        owner->storage = g.endpoint_storage();
        owner->graph = std::move(graph);
        const auto &edges = owner->storage->edges;
        const auto *data = edges.empty() ? &owner->empty_endpoint :
                           (source ? &edges.front().source : &edges.front().target);
        const auto size = static_cast<py::ssize_t>(edges.size());
        py::capsule base(owner.get(), [](void *p) {
            delete static_cast<endpoint_array_owner<graph_t> *>(p);
        });
        owner.release();
        py::array result(py::dtype::of<hg::index_t>(), {size},
                         {static_cast<py::ssize_t>(sizeof(typename graph_t::edge_descriptor))},
                         data, base);
        // The private capsule exposes no writable buffer, so NumPy cannot re-enable it.
        result.attr("setflags")(py::arg("write") = false);
        return result;
    }

    template<typename graph_t>
    struct def_add_edges {
        template<typename value_t, typename C>
        static
        void def(C &c, const char *doc) {
            c.def("add_edges", [](graph_t &g,
                                  pyarray<value_t> sources,
                                  pyarray<value_t> targets
                  ) {
                      hg_assert_vertex_indices(g, sources);
                      hg_assert_vertex_indices(g, targets);
                      hg::add_edges(sources, targets, g);
                  },
                  doc,
                  py::arg("sources"),
                  py::arg("targets"));
        }
    };

    template<typename graph_t, typename class_t>
    void init_graph(class_t &c) {

        using edge_t = typename hg::graph_traits<graph_t>::edge_descriptor;
        using vertex_t = typename hg::graph_traits<graph_t>::vertex_descriptor;
        using edge_index_t = typename hg::graph_traits<graph_t>::edge_index;

        c.def(py::init<const hg::size_t, const hg::size_t, const hg::size_t>(), R"doc(
    Create a new graph with no edge.

    :param number_of_vertices: initial number of vertices in the graph
    :param reserved_edges: pre-allocate space for the given number of edges
    :param reserved_edge_per_vertex: pre-allocate space for the given number of edge per vertex
    )doc",
              py::arg("number_of_vertices") = 0,
              py::arg("reserved_edges") = 0,
              py::arg("reserved_edge_per_vertex") = 0);

        add_edge_accessor_graph_concept<graph_t, decltype(c)>(c);
        add_incidence_graph_concept<graph_t, decltype(c)>(c);
        add_bidirectionnal_graph_concept<graph_t, decltype(c)>(c);
        add_adjacency_graph_concept<graph_t, decltype(c)>(c);
        add_vertex_list_graph_concept<graph_t, decltype(c)>(c);
        add_edge_list_graph_concept<graph_t, decltype(c)>(c);
        add_edge_index_graph_concept<graph_t, decltype(c)>(c);

        const char *endpoint_doc =
                "Zero-copy, read-only endpoint ndarray retaining its storage generation and graph. "
                "Length is fixed at creation; values may change in place or become stale after mutation. "
                "Obtain a new array for current topology, or copy before mutation for a snapshot.";
        c.def("_sources", [](py::object self) { return endpoint_array<graph_t>(std::move(self), true); },
              endpoint_doc);
        c.def("_targets", [](py::object self) { return endpoint_array<graph_t>(std::move(self), false); },
              endpoint_doc);

        c.def("add_edge", [](graph_t &g,
                             const vertex_t source,
                             const vertex_t target) {
                  hg_assert_vertex_index(g, source);
                  hg_assert_vertex_index(g, target);
                  return cpp_edge_2_python(hg::add_edge(source, target, g));
              },
              "Add an (undirected) edge between 'vertex1' and 'vertex2'. Returns the new edge.",
              py::arg("source"),
              py::arg("target"));

        add_type_overloads<def_add_edges<graph_t>, int, unsigned int, long long, unsigned long long>
                (c, "Add all edges given as a pair of arrays (sources, targets) to the graph.");

        c.def("add_vertex", [](graph_t &g) {
                  return hg::add_vertex(g);
              },
              "Add a vertex to the graph, the index of the new vertex is returned");
        c.def("add_vertices", [](graph_t &g, size_t num) {
                  hg::add_vertices(num, g);
              },
              py::arg("num"),
              "Add the given number of vertices to the graph.");
        c.def("set_edge", [](graph_t &g, index_t edge, index_t source, index_t target) {
                  hg_assert_edge_index(g, edge);
                  hg_assert_vertex_index(g, source);
                  hg_assert_vertex_index(g, target);
                  g.set_edge(edge, source, target);
              },
              py::arg("edge_index"),
              py::arg("source"),
              py::arg("target"),
              "Modify the source and the target of the given edge.");
        c.def("remove_edge", [](graph_t &g, index_t edge) {
                  hg_assert_edge_index(g, edge);
                  g.remove_edge(edge);
              },
              py::arg("edge_index"),
              "Remove the given edge from the graph (the edge is not really removed: "
              "its source and target are attached to a virtual node of index -1).");
    }

    void py_init_undirected_graph(py::module &m) {
        //xt::import_numpy();
        auto c = py::class_<hg::undirected_graph<hg::vecS> >(m,
                                                             "UndirectedGraph",
                                                             "A class to represent sparse undirected graph as adjacency lists.",
                                                             py::dynamic_attr());
        init_graph<hg::ugraph>(c);

        auto c2 = py::class_<hg::undirected_graph<hg::hash_setS>>(m, "UndirectedGraphOptimizedDelete");
        init_graph<hg::undirected_graph<hg::hash_setS >>(c2);
    }

}
