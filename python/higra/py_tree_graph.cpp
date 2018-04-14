//
// Created by user on 4/1/18.
//

#include "py_tree_graph.hpp"
#include "py_common_graph.hpp"
#include "xtensor-python/pyarray.hpp"


namespace py = pybind11;

using graph_t = hg::tree;
using edge_t = typename boost::graph_traits<graph_t>::edge_descriptor;
using vertex_t = typename boost::graph_traits<graph_t>::vertex_descriptor;


#define TREE_CTR(rawXKCD, dataXKCD, type) \
    c.def(py::init([](const xt::pyarray<type> & parent){return graph_t(parent);}), \
        "Create a tree from the given parent relation of type  " HG_XSTR(type) ".",    \
        py::arg("parentRelation")   \
        );


void py_init_tree_graph(pybind11::module &m) {
    xt::import_numpy();
    auto c = py::class_<graph_t>(m, "Tree");

    HG_FOREACH(TREE_CTR, HG_INTEGRAL_TYPES)

    add_incidence_graph_concept<graph_t, decltype(c)>(c);
    add_bidirectionnal_graph_concept<graph_t, decltype(c)>(c);
    add_adjacency_graph_concept<graph_t, decltype(c)>(c);
    add_vertex_list_graph_concept<graph_t, decltype(c)>(c);
    add_edge_list_graph_concept<graph_t, decltype(c)>(c);
    add_edge_index_graph_concept<graph_t, decltype(c)>(c);

    c.def("root", &graph_t::root, "Get the index of the root node (i.e. self.numVertices() - 1)");
    c.def("numLeaves", &graph_t::num_leaves, "Get the number of leaves nodes.");
    c.def("numChildren", &graph_t::num_children, "Get the number of children nodes of the given node.",
          py::arg("node"));
    c.def("children",
          [](const graph_t &g, vertex_t v) {
              auto it = hg::children(v, g);
              return pybind11::make_iterator(it.first, it.second);
          },
          "Get an iterator on the children of the given node.",
          py::arg("node"));
    c.def("parents", &graph_t::parents, "Get the parents array representing the tree.");
}

#undef TREE_CTR