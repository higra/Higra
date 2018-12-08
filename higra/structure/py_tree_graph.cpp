/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_tree_graph.hpp"
#include "py_common_graph.hpp"

namespace py = pybind11;

using graph_t = hg::tree;
using edge_t = graph_t::edge_descriptor;
using vertex_t = graph_t::vertex_descriptor;

template<typename graph_t>
struct def_tree_ctr {
    template<typename type, typename C>
    static
    void def(C &c, const char *doc) {
        c.def(py::init([](const pyarray<type> &parent) { return graph_t(parent); }),
              doc,
              py::arg("parent_relation")
        );
    }
};


void py_init_tree_graph(pybind11::module &m) {
    xt::import_numpy();
    auto c = py::class_<graph_t>(m, "Tree");

    add_type_overloads<def_tree_ctr<graph_t>, HG_TEMPLATE_INTEGRAL_TYPES>
            (c, "Create a tree from the given parent relation.");
    add_edge_accessor_graph_concept<graph_t, decltype(c)>(c);
    add_incidence_graph_concept<graph_t, decltype(c)>(c);
    add_bidirectionnal_graph_concept<graph_t, decltype(c)>(c);
    add_adjacency_graph_concept<graph_t, decltype(c)>(c);
    add_vertex_list_graph_concept<graph_t, decltype(c)>(c);
    add_edge_list_graph_concept<graph_t, decltype(c)>(c);
    add_edge_index_graph_concept<graph_t, decltype(c)>(c);

    c.def("root", &graph_t::root, "Get the index of the root node (i.e. self.num_vertices() - 1)");
    c.def("num_leaves", &graph_t::num_leaves, "Get the number of leaves nodes.");
    c.def("is_leaf", &graph_t::is_leaf, "Returns true if the given node is a leaf of true and false otherwise.");

    c.def("num_children", &graph_t::num_children, "Get the number of children nodes of the given node.",
          py::arg("node"));
    c.def("child", &graph_t::child, "Get the i-th (starting at 0) child of the given node of the tree.",
          py::arg("i"),
          py::arg("node"));
    c.def("children",
          [](const graph_t &g, vertex_t v) {
              auto it = hg::children(v, g);
              return pybind11::make_iterator(it.first, it.second);
          },
          "Get an iterator on the children of the given node.",
          py::arg("node"));
    c.def("parents", &graph_t::parents, "Get the parents array representing the tree.");
    c.def("parent", [](const graph_t &tree, vertex_t v) { return tree.parent(v); }, "Get the parent of the given node.",
          py::arg("node"));

    c.def("leaves_iterator",
          [](const graph_t &tree) {
              auto range = hg::leaves_iterator(tree);
              return py::make_iterator(range.begin(), range.end());
          }, "Returns an iterator on leaves of the tree.");

    c.def("leaves_to_root_iterator",
          [](const graph_t &tree, bool includeLeaves, bool includeRoot) {
              auto range = hg::leaves_to_root_iterator(tree,
                                                       (includeLeaves) ? hg::leaves_it::include
                                                                       : hg::leaves_it::exclude,
                                                       (includeRoot) ? hg::root_it::include : hg::root_it::exclude);
              return py::make_iterator(range.begin(), range.end());
          }, "Returns an iterator on the node indices going from the leaves to the root of the tree.",
          py::arg("include_leaves") = true,
          py::arg("include_root") = true);

    c.def("root_to_leaves_iterator",
          [](const graph_t &tree, bool includeLeaves, bool includeRoot) {
              auto range = root_to_leaves_iterator(tree,
                                                   (includeLeaves) ? hg::leaves_it::include : hg::leaves_it::exclude,
                                                   (includeRoot) ? hg::root_it::include : hg::root_it::exclude);
              return py::make_iterator(range.begin(), range.end());
          }, "Returns an iterator on the node indices going from the root to the leaves of the tree.",
          py::arg("include_leaves") = true,
          py::arg("include_root") = true);

}

