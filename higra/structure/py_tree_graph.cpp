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
struct def_out_degree_tree {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("out_degree", [](graph_t &g,
                               pyarray<value_t> vertices) {
                  g.compute_children();
                  hg_assert_vertex_indices(g, vertices);
                  return hg::out_degree(vertices, g);
              },
              doc,
              pybind11::arg("vertices_array"));
    }
};

template<typename graph_t>
struct def_degree_tree {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("degree", [](graph_t &g,
                           pyarray<value_t> vertices) {
                  g.compute_children();
                  hg_assert_vertex_indices(g, vertices);
                  return hg::degree(vertices, g);
              },
              doc,
              pybind11::arg("vertices_array"));
    }
};

template<typename graph_t>
struct def_in_degree_tree {
    template<typename value_t, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("in_degree", [](graph_t &g,
                              pyarray<value_t> vertices) {
                  g.compute_children();
                  hg_assert_vertex_indices(g, vertices);
                  return hg::in_degree(vertices, g);
              },
              doc,
              pybind11::arg("vertices_array"));
    }
};

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
                  hg_assert_vertex_indices(tree, vertices);
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
                  hg_assert_vertex_indices(tree, vertices);
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
                  hg_assert_vertex_index(tree, vertex);
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
        c.def("_num_children",
              [](const graph_t &tree, const pyarray<type> &vertices) {
                  hg_assert_vertex_indices(tree, vertices);
                  tree.compute_children();
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
        c.def("_child",
              [](const graph_t &tree, hg::index_t i, const pyarray<type> &vertices) {
                  tree.compute_children();
                  hg_assert_vertex_indices(tree, vertices);
                  hg_assert(i >= 0, "Child index cannot be negative.");
                  hg_assert(i < (index_t) (xt::amin)(num_children(vertices, tree))(),
                            "Child index is larger than the number of children.");
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
                  hg_assert_vertex_indices(tree, vertices);
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

    auto c = py::class_<graph_t>(m,
                                 "Tree",
                                 "An optimized static tree structure with nodes stored linearly in topological order (from leaves to root).",
                                 py::dynamic_attr());

    add_type_overloads<def_tree_ctr<graph_t>, HG_TEMPLATE_INTEGRAL_TYPES>
            (c, "Create a tree from the given parent relation.");
    add_edge_accessor_graph_concept<graph_t, decltype(c)>(c);

    ///////////////////////////////////
    // incidence graph concepts

    using iterator_transform_function = std::function<pybind11::tuple(edge_t)>;
    using out_edge_iterator = hg::transform_forward_iterator<iterator_transform_function,
            typename hg::graph_traits<graph_t>::out_edge_iterator,
            pybind11::tuple
    >;

    c.def("out_edges", [](const graph_t &g,
                          const vertex_t v) {
              hg_assert_vertex_index(g, v);
              g.compute_children();
              auto it = hg::out_edges(v, g);
              // wrapping out edge iterator to python friendly type
              auto it1 = out_edge_iterator(it.first, cpp_edge_2_python<edge_t>);
              auto it2 = out_edge_iterator(it.second, cpp_edge_2_python<edge_t>);
              return pybind11::make_iterator(it1, it2);

          },
          "Iterator over all out edges from 'vertex'. An out edge is a tuple '(vertex, adjacent_vertex)'.",
          pybind11::arg("vertex"));

    c.def("out_degree", [](graph_t &g, vertex_t vertex) {
              hg_assert_vertex_index(g, vertex);
              g.compute_children();
              return hg::out_degree(vertex, g);
          },
          "Return the out degree of the given vertex.",
          pybind11::arg("vertex"));

    add_type_overloads<def_out_degree_tree<graph_t>, int, unsigned int, long long, unsigned long long>
            (c, "Return the out degree of the given vertices.");

    ///////////////////////////////////
    // bidirectionnal graph concepts
    using in_edge_iterator = hg::transform_forward_iterator<
            iterator_transform_function,
            typename hg::graph_traits<graph_t>::in_edge_iterator,
            pybind11::tuple>;

    c.def("in_edges", [](graph_t &g,
                         const vertex_t v) {
              hg_assert_vertex_index(g, v);
              g.compute_children();
              auto it = hg::in_edges(v, g);
              // wrapping in edge iterator to python friendly type
              auto it1 = in_edge_iterator(it.first, cpp_edge_2_python<edge_t>);
              auto it2 = in_edge_iterator(it.second, cpp_edge_2_python<edge_t>);
              return pybind11::make_iterator(it1, it2);

          },
          "Iterator over all in edges from 'vertex'. An in edge is a tuple '(adjacent_vertex, vertex)'.",
          pybind11::arg("vertex"));

    c.def("degree", [](graph_t &g, vertex_t vertex) {
              hg_assert_vertex_index(g, vertex);
              g.compute_children();
              return hg::degree(vertex, g);
          },
          "Return the degree of the given vertex.",
          pybind11::arg("vertex"));

    add_type_overloads<def_degree_tree<graph_t>, int, unsigned int, long long, unsigned long long>
            (c, "Return the degree of the given vertices.");

    c.def("in_degree", [](graph_t &g, vertex_t vertex) {
              hg_assert_vertex_index(g, vertex);
              g.compute_children();
              return hg::in_degree(vertex, g);
          },
          "Return the in degree of the given vertex.",
          pybind11::arg("vertex"));

    add_type_overloads<def_in_degree_tree<graph_t>, int, unsigned int, long long, unsigned long long>
            (c, "Return the in degree of the given vertices.");

    ///////////////////////////////////
    // adjacency graph concepts

    c.def("adjacent_vertices", [](const graph_t &g,
                                  const vertex_t v) {
              g.compute_children();
              hg_assert_vertex_index(g, v);
              auto it = hg::adjacent_vertices(v, g);
              return pybind11::make_iterator(it.first, it.second);
          },
          "Iterator over all vertices adjacent to the given vertex.",
          pybind11::arg("vertex"));

    add_vertex_list_graph_concept<graph_t, decltype(c)>(c);
    add_edge_list_graph_concept<graph_t, decltype(c)>(c);
    add_edge_index_graph_concept<graph_t, decltype(c)>(c);

    c.def("category", &graph_t::category, "Get the tree category (see enumeration TreeCategory)");
    c.def("root", &graph_t::root, "Get the index of the root node (i.e. self.num_vertices() - 1)");
    c.def("num_leaves", &graph_t::num_leaves, "Get the number of leaves nodes.");
    c.def("is_leaf", [](const graph_t &t, index_t i) {
              hg_assert_vertex_index(t, i);
              return t.is_leaf(i);
          },
          "Returns true if the given node is a leaf of true and false otherwise.",
          py::arg("vertex"));

    add_type_overloads<def_is_leaf<graph_t>, int, unsigned int, long long, unsigned long long>
            (c, "Indicates if each vertex in the given array is a leaf or not.");

    c.def("_num_children", [](const graph_t &t, index_t i) {
        hg_assert_vertex_index(t, i);
        t.compute_children();
        return t.num_children(i);
    }, "", py::arg("vertex"));
    add_type_overloads<def_num_children<graph_t>, int, unsigned int, long long, unsigned long long>(c, "");

    c.def("_child", [](const graph_t &t, index_t i, index_t vertex) {
              t.compute_children();
              hg_assert_vertex_index(t, vertex);
              hg_assert(i >= 0, "Child index cannot be negative.");
              hg_assert(i < (index_t) t.num_children(vertex),
                        "Child index is larger than the number of children.");
              return t.child(i, vertex);
          }, "Get the i-th (starting at 0) child of the given node of the tree.",
          py::arg("i"),
          py::arg("node"));
    add_type_overloads<def_child<graph_t>, int, unsigned int, long long, unsigned long long>
            (c, "Get the i-th (starting at 0) child of each vertex in the given array.");
    add_type_overloads<def_find_region<graph_t>, HG_TEMPLATE_NUMERIC_TYPES>
            (c,
             "Get largest vertex which contains the given vertex and whose altitude is stricly less than the given altitude lambda.");

    c.def("_lowest_common_ancestor", [](const graph_t &tree, hg::index_t vertex1, hg::index_t vertex2) {
              hg_assert_vertex_index(tree, vertex1);
              hg_assert_vertex_index(tree, vertex2);
              return hg::lowest_common_ancestor(vertex1, vertex2, tree);
          },
          "Return the lowest common ancestor of `vertex1` and `vertex2`. Worst case complexity is linear `O(N)`: consider using the `LCAFast` if many lowest common ancestors are needed",
          py::arg("vertex1"),
          py::arg("vertex2"));

    c.def("_lowest_common_ancestor",
          [](const graph_t &tree, const pyarray<hg::index_t> &vertices1, const pyarray<hg::index_t> &vertices2) {
              hg_assert_vertex_indices(tree, vertices1);
              hg_assert_vertex_indices(tree, vertices2);
              return hg::lowest_common_ancestor(vertices1, vertices2, tree);
          }, "Return the lowest common ancestor between any pairs of vertices in `vertices1` and `vertices2`.\n"
             "`vertices1` and `vertices2` must be 1d arrays of integers of the same size K"
             "Worst case complexity is `O(N*K)`: consider using the `LCAFast` if many lowest common ancestors are needed",
          py::arg("vertices1"),
          py::arg("vertices2"));

    c.def("children",
          [](const graph_t &g, vertex_t v) {
              hg_assert_vertex_index(g, v);
              g.compute_children();
              hg::array_1d<hg::index_t> a = hg::array_1d<hg::index_t>::from_shape({hg::num_children(v, g)});
              auto it = hg::children(v, g);
              std::copy(it.first, it.second, a.begin());
              return a;
          },
          "Get a copy of the list of children of the given node.",
          py::arg("node"));

    c.def("parents",
          &graph_t::parents,
          "Get the parents array representing the tree.",
          pybind11::return_value_policy::reference_internal);
    c.def("parent", [](const graph_t &tree, vertex_t v) {
              hg_assert_vertex_index(tree, v);
              return tree.parent(v);
          },
          "Get the parent of the given node.",
          py::arg("node"));
    add_type_overloads<def_parent<graph_t>, int, unsigned int, long long, unsigned long long>
            (c, "Get the parent of each vertex in the given array.");

    c.def("ancestors", [](const graph_t &tree, vertex_t v) {
              hg_assert_vertex_index(tree, v);
              std::vector<hg::index_t> vec;
              for (auto c: hg::ancestors_iterator(v, tree)) {
                  vec.push_back(c);
              }
              hg::array_1d<hg::index_t> a = hg::array_1d<hg::index_t>::from_shape({vec.size()});
              std::copy(vec.begin(), vec.end(), a.begin());
              return a;
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

    c.def("_compute_children", &graph_t::compute_children, "Compute the children relation.");
    c.def("_children_computed", &graph_t::children_computed,
          "True if the children relation has already been computed.");
    c.def("clear_children", &graph_t::clear_children, "Remove the children relation if it has already been computed. "
                                                      "May free memory but only useful if you are sure that this relation won't be required by further processing).");
}

