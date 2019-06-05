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
        c.def(py::init(
                [](const pyarray<type> &parent, hg::tree_category category) { return graph_t(parent, category); }),
              doc,
              py::arg("parent_relation"),
              py::arg("category") = hg::tree_category::partition_tree
        );
    }
};

template<typename graph_t>
struct def_is_leaf {
    template<typename type, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("is_leaf",
              [](const graph_t &tree, const pyarray<type> &vertices) {
                  return hg::is_leaf(vertices, tree);
              },
              doc,
              py::arg("vertices")
        );
    }
};

template<typename graph_t>
struct def_find_region {
    template<typename type, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_find_region",
              [](const graph_t &tree,
                 const pyarray<hg::index_t> &vertices,
                 const pyarray<type> &lambdas,
                 const pyarray<type> &altitudes) {
                  return hg::find_region(vertices, lambdas, altitudes, tree);
              },
              doc,
              py::arg("vertices"),
              py::arg("lambdas"),
              py::arg("altitudes")
        );
        c.def("_find_region",
              [](const graph_t &tree,
                 hg::index_t vertex,
                 type lambda,
                 const pyarray<type> &altitudes) {
                  return hg::find_region(vertex, lambda, altitudes, tree);
              },
              doc,
              py::arg("vertex"),
              py::arg("lambda"),
              py::arg("altitudes")
        );
    }
};


template<typename graph_t>
struct def_num_children {
    template<typename type, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("num_children",
              [](const graph_t &tree, const pyarray<type> &vertices) {
                  return hg::num_children(vertices, tree);
              },
              doc,
              py::arg("vertices")
        );
    }
};

template<typename graph_t>
struct def_child {
    template<typename type, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("child",
              [](const graph_t &tree, hg::index_t i, const pyarray<type> &vertices) {
                  return hg::child(i, vertices, tree);
              },
              doc,
              py::arg("i"),
              py::arg("vertices")
        );
    }
};

template<typename graph_t>
struct def_parent {
    template<typename type, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("parent",
              [](const graph_t &tree, const pyarray<type> &vertices) {
                  return hg::parent(vertices, tree);
              },
              doc,
              py::arg("vertices")
        );
    }
};

void py_init_tree_graph(pybind11::module &m) {
    xt::import_numpy();

    py::enum_<hg::tree_category>(m, "TreeCategory",
                                 "Category of hierarchies.")
            .value("ComponentTree", hg::tree_category::component_tree)
            .value("PartitionTree", hg::tree_category::partition_tree);

    auto c = py::class_<graph_t>(m, "Tree",
            "An optimized static tree structure with nodes stored linearly in topological order (from leaves to root).");

    add_type_overloads<def_tree_ctr<graph_t>, HG_TEMPLATE_INTEGRAL_TYPES>
            (c, "Create a tree from the given parent relation.");
    add_edge_accessor_graph_concept<graph_t, decltype(c)>(c);
    add_incidence_graph_concept<graph_t, decltype(c)>(c);
    add_bidirectionnal_graph_concept<graph_t, decltype(c)>(c);
    add_adjacency_graph_concept<graph_t, decltype(c)>(c);
    add_vertex_list_graph_concept<graph_t, decltype(c)>(c);
    add_edge_list_graph_concept<graph_t, decltype(c)>(c);
    add_edge_index_graph_concept<graph_t, decltype(c)>(c);

    c.def("category", &graph_t::category, "Get the tree category (see enumeration TreeCategory)");
    c.def("root", &graph_t::root, "Get the index of the root node (i.e. self.num_vertices() - 1)");
    c.def("num_leaves", &graph_t::num_leaves, "Get the number of leaves nodes.");
    c.def("is_leaf", &graph_t::is_leaf, "Returns true if the given node is a leaf of true and false otherwise.");
    add_type_overloads<def_is_leaf<graph_t>, int, unsigned int, long long, unsigned long long>
            (c, "Indicates if each vertex in the given array is a leaf or not.");

    c.def("num_children", &graph_t::num_children, "Get the number of children nodes of the given node.",
          py::arg("node"));
    add_type_overloads<def_num_children<graph_t>, int, unsigned int, long long, unsigned long long>
            (c, "Get the number of children of each vertex in the given array.");

    c.def("child", &graph_t::child, "Get the i-th (starting at 0) child of the given node of the tree.",
          py::arg("i"),
          py::arg("node"));
    add_type_overloads<def_child<graph_t>, int, unsigned int, long long, unsigned long long>
            (c, "Get the i-th (starting at 0) child of each vertex in the given array.");
    add_type_overloads<def_find_region<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (c,
             "Get largest vertex which contains the given vertex and whose altitude is stricly less than the given altitude lambda.");

    c.def("lowest_common_ancestor", [](const graph_t& tree, hg::index_t vertex1, hg::index_t vertex2){
        return hg::lowest_common_ancestor(vertex1, vertex2, tree);
        }, "Return the lowest common ancestor of `vertex1` and `vertex2`. Worst case complexity is linear `O(N)`: consider using the `LCAFast` if many lowest common ancestors are needed",
          py::arg("vertex1"),
          py::arg("vertex2"));

    c.def("lowest_common_ancestor", [](const graph_t& tree, const pyarray<hg::index_t>& vertices1, const pyarray<hg::index_t>& vertices2){
              return hg::lowest_common_ancestor(vertices1, vertices2, tree);
          }, "Return the lowest common ancestor between any pairs of vertices in `vertices1` and `vertices2`.\n"
             "`vertices1` and `vertices2` must be 1d arrays of integers of the same size K"
             "Worst case complexity is `O(N*K)`: consider using the `LCAFast` if many lowest common ancestors are needed",
          py::arg("vertices1"),
          py::arg("vertices2"));
    
    c.def("children",
          [](const graph_t &g, vertex_t v) {
              pybind11::list l;
              for (auto c: hg::children_iterator(v, g)) {
                  l.append(c);
              }
              return l;
          },
          "Get a copy of the list of children of the given node.",
          py::arg("node"));

    c.def("parents", &graph_t::parents, "Get the parents array representing the tree.");
    c.def("parent", [](const graph_t &tree, vertex_t v) { return tree.parent(v); }, "Get the parent of the given node.",
          py::arg("node"));
    add_type_overloads<def_parent<graph_t>, int, unsigned int, long long, unsigned long long>
            (c, "Get the parent of each vertex in the given array.");

    c.def("ancestors", [](const graph_t &tree, vertex_t v) {
              pybind11::list l;
              for (auto c: hg::ancestors_iterator(v, tree)) {
                  l.append(c);
              }
              return l;
          },
          "Get the list of ancestors of the given node in topological order (starting from the given node included).",
          py::arg("node"));

    c.def("leaves",
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

