/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_binary_partition_tree.hpp"

#include "py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "higra/hierarchy/binary_partition_tree.hpp"

template<typename T>
using pyarray = xt::pyarray<T>;

using namespace hg;
namespace py = pybind11;

template<typename T>
void def_new_neighbour(pybind11::module &m) {
    using class_t = typename hg::binary_partition_tree_internal::new_neighbour<T>;
    auto c = py::class_<class_t>(m, (std::string("NewNeighbour_") + typeid(T).name()).c_str(),
                                 "This structure is provided by the binary partition algorithm "
                                 "when two nodes are merged in order to compute the edge weight between the newly created node and one "
                                 "of its neighbouring node.");
    c.def("num_edges", &class_t::num_edges,
          "Number of edges between the merged nodes and the neighbour node");
    c.def("first_edge_index", [](const class_t &c) { return c.first_edge_index(); },
          "The index of the edge linking the first merged node with the neighbouring node.");
    c.def("second_edge_index", [](const class_t &c) { return c.second_edge_index(); },
          "The index of the edge linking the second merged node with the neighbouring node (-1 if num_edges() < 2).");
    c.def("neighbour_vertex", [](const class_t &c) { return c.neighbour_vertex(); },
          "The index of the neighbour node.");
    c.def("set_new_edge_weight", [](class_t &e, T value) { e.new_edge_weight() = value; },
          "The new value of the edge linking the new node to the new neighbour.");
    c.def("new_edge_index", [](const class_t &c) { return c.new_edge_index(); },
          "The index of the edge linking the new node to the neighbour node.");
}

struct def_binary_partition_tree_average_linkage {
    template<typename T>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("_binary_partition_tree_average_linkage",
              [](const hg::ugraph &graph, pyarray<T> &edge_values, pyarray<T> &edge_weights) {
                  return hg::binary_partition_tree(graph,
                                                   edge_values,
                                                   hg::make_binary_partition_tree_average_linkage(
                                                           edge_values, edge_weights));
              },
              doc,
              py::arg("graph"),
              py::arg("edge_values"),
              py::arg("edge_weights"));
    }
};

struct def_binary_partition_tree_custom_linkage {
    template<typename T>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("_binary_partition_tree",
              [](const hg::ugraph &graph,
                 pyarray<T> &edge_weights,
                 py::object weighting_function) {
                  using new_neighbours_type = const std::vector<binary_partition_tree_internal::new_neighbour<T> >;
                  auto weighter = [&weighting_function](
                          const undirected_graph<hg::undirected_graph_internal::hash_setS> &g,
                          index_t fusion_edge_index,
                          index_t new_region,
                          index_t merged_region1,
                          index_t merged_region2,
                          new_neighbours_type &new_neighbours) {
                      weighting_function(g, fusion_edge_index, new_region, merged_region1, merged_region2,
                                         pybind11::make_iterator(new_neighbours.begin(), new_neighbours.end()));
                  };
                  return hg::binary_partition_tree(graph,
                                                   edge_weights,
                                                   weighter);
              },
              doc,
              py::arg("graph"),
              py::arg("edge_weights"),
              py::arg("weighting_function"));
    }
};

struct def_binary_partition_tree_complete_linkage {
    template<typename T>
    static
    void def(pybind11::module &m, const char *doc) {
        m.def("_binary_partition_tree_complete_linkage",
              [](const hg::ugraph &graph, pyarray<T> &edge_weights) {
                  return hg::binary_partition_tree(graph,
                                                   edge_weights,
                                                   hg::make_binary_partition_tree_complete_linkage(
                                                           edge_weights));
              },
              doc,
              py::arg("graph"),
              py::arg("edge_weights"));
    }
};

void py_init_binary_partition_tree(pybind11::module &m) {
    xt::import_numpy();


    add_type_overloads<def_binary_partition_tree_average_linkage, HG_TEMPLATE_FLOAT_TYPES>
            (m,
             "Compute a binary partition tree with average linkage distance.\n"
             "\n"
             "Given a graph G, with initial edge values V with associated weights W,\n"
             "the distance d(X,Y) between any two regions X, Y is defined as :\n"
             "d(X,Y) = (1 / Z) + sum_{x in X, y in Y, {x,y} in G} V({x,y}) x W({x,y})\n"
             "with Z = sum_{x in X, y in Y, {x,y} in G} W({x,y})");
    add_type_overloads<def_binary_partition_tree_complete_linkage, HG_TEMPLATE_FLOAT_TYPES>
            (m,
             "Compute a binary partition tree with complete linkage distance.\n"
             "\n"
             "Given a graph G, with initial edge weights W,\n"
             "the distance d(X,Y) between any two regions X, Y is defined as :\n"
             "d(X,Y) = max {W({x,y}) | x in X, y in Y, {x,y} in G }");

    def_new_neighbour<float>(m);
    def_new_neighbour<double>(m);
    add_type_overloads<def_binary_partition_tree_custom_linkage, HG_TEMPLATE_FLOAT_TYPES>
            (m,
             "Compute a binary partition tree with the given linkage distance.");
}
